#include "fqi_teacher.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <thread>
#include <unordered_map>

namespace g001::fqi {
namespace {

std::size_t align_up(std::size_t x, std::size_t alignment) {
    return (x + alignment - 1) & ~(alignment - 1);
}
std::uint64_t mix(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
bool valid(const Transition& row, std::size_t option) { return (row.valid_option_mask >> option) & 1ULL; }
bool done(const Transition& row, std::size_t option) { return (row.done_option_mask >> option) & 1ULL; }

float predict_tree(const RegressionTree& tree, const std::vector<float>& feature) {
    if (tree.nodes.empty()) return 0;
    int node = 0;
    for (std::size_t guard = 0; guard <= tree.nodes.size(); ++guard) {
        const auto& n = tree.nodes.at(node);
        if (n.feature < 0) return n.value;
        node = feature.at(n.feature) <= n.threshold ? n.left : n.right;
    }
    return 0;
}
float predict_forest(const Forest& forest, const std::vector<float>& feature) {
    if (forest.trees.empty()) return 0;
    double sum = 0;
    for (const auto& tree : forest.trees) sum += predict_tree(tree, feature);
    return static_cast<float>(sum / forest.trees.size());
}

class Dsu {
 public:
    explicit Dsu(std::size_t n) : p_(n), size_(n, 1) { std::iota(p_.begin(), p_.end(), 0); }
    std::size_t find(std::size_t x) { return p_[x] == x ? x : p_[x] = find(p_[x]); }
    void join(std::size_t a, std::size_t b) {
        a = find(a); b = find(b); if (a == b) return;
        if (size_[a] < size_[b]) std::swap(a, b);
        p_[b] = a; size_[a] += size_[b];
    }
 private: std::vector<std::size_t> p_, size_;
};

struct Partition {
    std::vector<std::size_t> train, holdout;
    std::vector<std::vector<std::size_t>> train_components;
    std::size_t components{};
};

Partition partition(const TransitionDataset& data, const FqiConfig& cfg) {
    Dsu dsu(data.rows.size());
    std::unordered_map<std::uint64_t, std::size_t> episodes, groups;
    for (std::size_t i = 0; i < data.rows.size(); ++i) {
        auto [e, inserted] = episodes.emplace(data.rows[i].episode_id, i);
        if (!inserted) dsu.join(i, e->second);
        if (data.rows[i].split_group) {
            auto [g, fresh] = groups.emplace(data.rows[i].split_group, i);
            if (!fresh) dsu.join(i, g->second);
        }
        for (std::size_t a = 0; a < data.option_count; ++a) {
            if (!valid(data.rows[i], a) || done(data.rows[i], a)) continue;
            const auto next = data.rows[i].next_state[a];
            if (next < 0 || static_cast<std::size_t>(next) >= data.rows.size())
                throw std::runtime_error("nonterminal option has invalid next_state");
            dsu.join(i, static_cast<std::size_t>(next));
        }
    }
    std::unordered_map<std::size_t, std::vector<std::size_t>> map;
    for (std::size_t i = 0; i < data.rows.size(); ++i) map[dsu.find(i)].push_back(i);
    if (map.size() < 2) throw std::runtime_error("FQI grouped holdout needs >=2 disconnected components");
    std::vector<std::vector<std::size_t>> components;
    for (auto& [root, rows] : map) { (void)root; components.push_back(std::move(rows)); }
    Partition out; out.components = components.size();
    for (auto& rows : components) {
        const auto& first = data.rows[rows.front()];
        const double unit = static_cast<double>(mix(first.episode_id ^ mix(first.split_group) ^ cfg.seed) >> 11) /
                            static_cast<double>(1ULL << 53);
        if (unit < cfg.holdout_fraction) out.holdout.insert(out.holdout.end(), rows.begin(), rows.end());
        else { out.train.insert(out.train.end(), rows.begin(), rows.end()); out.train_components.push_back(rows); }
    }
    if (out.holdout.empty()) {
        out.holdout = out.train_components.back(); out.train_components.pop_back();
        std::vector<bool> moved(data.rows.size()); for (auto r : out.holdout) moved[r] = true;
        out.train.erase(std::remove_if(out.train.begin(), out.train.end(), [&](auto r){return moved[r];}), out.train.end());
    }
    if (out.train.empty() || out.train_components.empty()) throw std::runtime_error("empty grouped FQI training split");
    return out;
}

class RegressionBuilder {
 public:
    RegressionBuilder(const TransitionDataset& data, const std::vector<float>& target,
                      const FqiConfig& cfg, std::mt19937_64& rng)
        : data_(data), target_(target), cfg_(cfg), rng_(rng) {}
    RegressionTree fit(const std::vector<std::size_t>& rows) { build(rows, 0); return std::move(tree_); }
 private:
    int build(const std::vector<std::size_t>& rows, int depth) {
        double sum = 0; for (auto r : rows) sum += target_[r];
        const float mean = rows.empty() ? 0 : static_cast<float>(sum / rows.size());
        const int index = static_cast<int>(tree_.nodes.size());
        tree_.nodes.push_back({.value=mean});
        if (depth >= cfg_.tree_depth || rows.size() < 2 * cfg_.min_leaf) return index;
        std::vector<int> features(data_.feature_count); std::iota(features.begin(), features.end(), 0);
        std::shuffle(features.begin(), features.end(), rng_);
        const auto count = std::min<std::size_t>(features.size(), cfg_.random_features ? cfg_.random_features :
                                                 std::max<std::size_t>(1, std::sqrt(features.size())));
        double parent_sse = 0; for (auto r : rows) { double d=target_[r]-mean; parent_sse+=d*d; }
        double best_sse = parent_sse; int best_feature=-1; float best_threshold=0;
        for (std::size_t fi=0; fi<count; ++fi) {
            const int f=features[fi]; float lo=data_.rows[rows[0]].feature[f], hi=lo;
            for(auto r:rows){lo=std::min(lo,data_.rows[r].feature[f]);hi=std::max(hi,data_.rows[r].feature[f]);}
            if (!(lo<hi)) continue;
            std::uniform_real_distribution<float> threshold(lo,hi);
            for(std::size_t k=0;k<cfg_.random_thresholds;++k){
                const float t=threshold(rng_); double ls=0,ls2=0,rs=0,rs2=0;std::size_t ln=0,rn=0;
                for(auto r:rows){double y=target_[r];if(data_.rows[r].feature[f]<=t){ls+=y;ls2+=y*y;++ln;}else{rs+=y;rs2+=y*y;++rn;}}
                if(ln<cfg_.min_leaf||rn<cfg_.min_leaf)continue;
                const double sse=(ls2-ls*ls/ln)+(rs2-rs*rs/rn);
                if(sse<best_sse){best_sse=sse;best_feature=f;best_threshold=t;}
            }
        }
        if(best_feature<0||parent_sse-best_sse<1e-9)return index;
        std::vector<std::size_t> left,right;left.reserve(rows.size());right.reserve(rows.size());
        for(auto r:rows)(data_.rows[r].feature[best_feature]<=best_threshold?left:right).push_back(r);
        const int l=build(left,depth+1), rr=build(right,depth+1);
        auto& n=tree_.nodes[index];n.feature=best_feature;n.threshold=best_threshold;n.left=l;n.right=rr;
        return index;
    }
    const TransitionDataset& data_; const std::vector<float>& target_; const FqiConfig& cfg_;
    std::mt19937_64& rng_; RegressionTree tree_;
};

struct FitResult { Teacher teacher; std::vector<IterationMetric> convergence; };

FitResult fit_teacher(const TransitionDataset& data, const std::vector<std::size_t>& rows,
                      const std::vector<std::vector<std::size_t>>& components,
                      const FqiConfig& cfg, std::uint64_t seed) {
    Teacher previous; previous.option.resize(data.option_count);
    FitResult result;
    for (std::size_t iteration=0; iteration<cfg.iterations; ++iteration) {
        std::vector<std::vector<float>> targets(data.option_count, std::vector<float>(data.rows.size()));
        for(auto r:rows)for(std::size_t a=0;a<data.option_count;++a)if(valid(data.rows[r],a)){
            double y=data.rows[r].reward[a];
            if(!done(data.rows[r],a)){
                const auto next=static_cast<std::size_t>(data.rows[r].next_state[a]);
                auto q=previous.q_values(data.rows[next].feature);double best=-std::numeric_limits<double>::infinity();
                for(std::size_t b=0;b<data.option_count;++b)if(valid(data.rows[next],b))best=std::max(best,static_cast<double>(q[b]));
                if(std::isfinite(best))y+=best; // gamma=1; duration is SMDP timing metadata
            }
            targets[a][r]=static_cast<float>(y);
        }
        Teacher current;current.option.resize(data.option_count);
        for(auto& forest:current.option)forest.trees.resize(cfg.trees);
        std::atomic<std::size_t> next_task{0};const std::size_t tasks=data.option_count*cfg.trees;
        const auto threads=std::min<std::size_t>(tasks,cfg.threads?cfg.threads:std::max(1u,std::thread::hardware_concurrency()));
        std::vector<std::thread> workers;
        for(std::size_t w=0;w<threads;++w)workers.emplace_back([&]{for(;;){
            const auto task=next_task.fetch_add(1);if(task>=tasks)break;const auto a=task/cfg.trees, ti=task%cfg.trees;
            std::mt19937_64 rng(mix(seed^(iteration<<40)^(a<<24)^ti));
            std::uniform_int_distribution<std::size_t> choose(0,components.size()-1);std::vector<std::size_t> sample;
            for(std::size_t draw=0;draw<components.size();++draw)for(auto r:components[choose(rng)])if(valid(data.rows[r],a))sample.push_back(r);
            if(sample.empty())for(auto r:rows)if(valid(data.rows[r],a))sample.push_back(r);
            if(sample.empty())throw std::runtime_error("simulator supplied no coverage for option "+std::to_string(a));
            current.option[a].trees[ti]=RegressionBuilder(data,targets[a],cfg,rng).fit(sample);
        }});for(auto& worker:workers)worker.join();
        double squared=0,max_change=0;std::size_t n=0;
        for(auto r:rows){auto oldq=previous.q_values(data.rows[r].feature),newq=current.q_values(data.rows[r].feature);
            for(std::size_t a=0;a<data.option_count;++a)if(valid(data.rows[r],a)){
                double target=data.rows[r].reward[a];
                if(!done(data.rows[r],a)){
                    const auto next=static_cast<std::size_t>(data.rows[r].next_state[a]);auto nq=current.q_values(data.rows[next].feature);double best=-std::numeric_limits<double>::infinity();
                    for(std::size_t b=0;b<data.option_count;++b)if(valid(data.rows[next],b))best=std::max(best,static_cast<double>(nq[b]));
                    target+=best;
                }
                const double d=newq[a]-target;squared+=d*d;++n;max_change=std::max(max_change,static_cast<double>(std::abs(newq[a]-oldq[a])));}}
        result.convergence.push_back({iteration+1,n?std::sqrt(squared/n):0,max_change});
        previous=std::move(current);
    }
    result.teacher=std::move(previous);return result;
}

std::size_t greedy(const Teacher& t,const Transition& row){auto q=t.q_values(row.feature);std::size_t best=0;float value=-std::numeric_limits<float>::infinity();for(std::size_t a=0;a<q.size();++a)if(valid(row,a)&&q[a]>value){value=q[a];best=a;}return best;}

double bellman_rmse(const Teacher& teacher,const TransitionDataset& data,const std::vector<std::size_t>& rows){double sq=0;std::size_t n=0;for(auto r:rows){auto q=teacher.q_values(data.rows[r].feature);for(std::size_t a=0;a<data.option_count;++a)if(valid(data.rows[r],a)){double target=data.rows[r].reward[a];if(!done(data.rows[r],a)){auto nq=teacher.q_values(data.rows[data.rows[r].next_state[a]].feature);double best=-1e300;for(std::size_t b=0;b<data.option_count;++b)if(valid(data.rows[data.rows[r].next_state[a]],b))best=std::max(best,static_cast<double>(nq[b]));target+=best;}double d=q[a]-target;sq+=d*d;++n;}}return n?std::sqrt(sq/n):0;}

std::vector<std::size_t> roots(const TransitionDataset& data,const std::vector<std::size_t>& rows){std::vector<bool> inside(data.rows.size()),referenced(data.rows.size());for(auto r:rows)inside[r]=true;for(auto r:rows)for(std::size_t a=0;a<data.option_count;++a)if(valid(data.rows[r],a)&&!done(data.rows[r],a)){auto n=data.rows[r].next_state[a];if(n>=0&&inside[n])referenced[n]=true;}std::vector<std::size_t> out;for(auto r:rows)if(!referenced[r])out.push_back(r);return out;}

double rollout(const Teacher* teacher,const TransitionDataset& data,std::size_t start,int mode,std::vector<double>* memo=nullptr,std::vector<int>* visiting=nullptr){
    if(mode==2&&(*memo)[start]==(*memo)[start])return(*memo)[start];
    if(mode==2){if((*visiting)[start])throw std::runtime_error("transition graph cycle with gamma=1");(*visiting)[start]=1;}
    const auto& row=data.rows[start];double best=-1e300;std::size_t selected=0;
    if(mode==0)selected=valid(row,0)?0:static_cast<std::size_t>(std::countr_zero(row.valid_option_mask));
    else if(mode==1)selected=greedy(*teacher,row);
    for(std::size_t a=0;a<data.option_count;++a)if(valid(row,a)){
        if(mode!=2&&a!=selected)continue;
        double value=row.reward[a];if(!done(row,a))value+=rollout(teacher,data,row.next_state[a],mode,memo,visiting);if(value>best)best=value;}
    if(mode==2){(*visiting)[start]=0;(*memo)[start]=best;}return best;
}
double policy_value(const Teacher* teacher,const TransitionDataset& data,const std::vector<std::size_t>& rows,int mode){auto start=roots(data,rows);if(start.empty())return 0;std::vector<double> memo(data.rows.size(),std::numeric_limits<double>::quiet_NaN());std::vector<int> visiting(data.rows.size());double sum=0;for(auto r:start)sum+=rollout(teacher,data,r,mode,&memo,&visiting);return sum/start.size();}

struct DistillBuilder {
    const TransitionDataset& data;const Teacher& teacher;std::vector<std::vector<float>> q;int max_depth;std::size_t min_leaf;double alpha;OptionTree tree;
    int build(const std::vector<std::size_t>& rows,int depth){std::size_t options=data.option_count,best_a=0;double leaf=-1e300;for(std::size_t a=0;a<options;++a){double v=0;bool ok=true;for(auto r:rows){if(!valid(data.rows[r],a)){ok=false;break;}v+=q[r][a];}if(ok&&v>leaf){leaf=v;best_a=a;}}int index=tree.nodes.size();tree.nodes.push_back({.option_id=static_cast<std::uint32_t>(best_a),.samples=static_cast<std::uint32_t>(rows.size())});if(depth>=max_depth||rows.size()<2*min_leaf)return index;int bf=-1;float bt=0;double bv=leaf;for(std::size_t f=0;f<data.feature_count;++f){std::vector<std::pair<float,std::size_t>> order;for(auto r:rows)order.push_back({data.rows[r].feature[f],r});std::sort(order.begin(),order.end());for(std::size_t i=min_leaf;i+min_leaf<=order.size();i+=std::max<std::size_t>(1,order.size()/64)){if(order[i-1].first==order[i].first)continue;double total=0;bool possible=true;for(int side=0;side<2;++side){double sidebest=-1e300;for(std::size_t a=0;a<options;++a){double v=0;bool ok=true;auto begin=side?i:0,end=side?order.size():i;for(auto k=begin;k<end;++k){auto r=order[k].second;if(!valid(data.rows[r],a)){ok=false;break;}v+=q[r][a];}if(ok)sidebest=std::max(sidebest,v);}if(!std::isfinite(sidebest)){possible=false;break;}total+=sidebest;}if(possible&&total>bv){bv=total;bf=f;bt=std::midpoint(order[i-1].first,order[i].first);}}}if(bf<0||(bv-leaf)<=alpha*rows.size())return index;std::vector<std::size_t>l,r;for(auto row:rows)(data.rows[row].feature[bf]<=bt?l:r).push_back(row);auto checkpoint=tree.nodes.size();int li=build(l,depth+1),ri=build(r,depth+1);if(bv-leaf<=alpha*rows.size()){tree.nodes.resize(checkpoint);return index;}auto&n=tree.nodes[index];n.feature=bf;n.threshold=bt;n.left=li;n.right=ri;return index;}
};

double quantile(std::vector<double> v,double q){if(v.empty())return 0;std::sort(v.begin(),v.end());double p=q*(v.size()-1);auto l=static_cast<std::size_t>(p),h=static_cast<std::size_t>(std::ceil(p));return v[l]+(v[h]-v[l])*(p-l);}

} // namespace

RowLayout row_layout(std::size_t features,std::size_t options){if(!options||options>64)throw std::runtime_error("option_count must be 1..64");RowLayout l;l.reward_offset=sizeof(TransitionRowPrefix);l.duration_offset=l.reward_offset+4*options;l.next_offset=align_up(l.duration_offset+2*options,alignof(std::int64_t));l.feature_offset=l.next_offset+8*options;l.row_size=l.feature_offset+4*features;return l;}

TransitionDataset load_transitions(const std::string& path){std::ifstream in(path,std::ios::binary);TransitionHeader h{};in.read(reinterpret_cast<char*>(&h),sizeof(h));if(!in||std::memcmp(h.magic,"MFQITRN1",8)||h.version!=1||h.gamma!=1.0f||h.reward_semantics!=1)throw std::runtime_error("invalid FQI transition header");auto layout=row_layout(h.feature_count,h.option_count);if(h.row_size!=layout.row_size)throw std::runtime_error("FQI row ABI mismatch");TransitionDataset d;d.base_feature_count=h.feature_count;d.feature_count=h.feature_count+9;d.option_count=h.option_count;d.option_schema_hash=h.option_schema_hash;d.rows.reserve(h.row_count);std::vector<char> bytes(h.row_size);for(std::uint64_t i=0;i<h.row_count;++i){in.read(bytes.data(),bytes.size());if(!in)throw std::runtime_error("truncated FQI row");TransitionRowPrefix p{};std::memcpy(&p,bytes.data(),sizeof(p));if(p.product_id>=9)throw std::runtime_error("product_id must be a tradable product 0..8");Transition r;r.episode_id=p.episode_id;r.split_group=p.split_group;r.valid_option_mask=p.valid_option_mask;r.done_option_mask=p.done_option_mask;r.product_id=p.product_id;r.reward.resize(h.option_count);r.duration.resize(h.option_count);r.next_state.resize(h.option_count);r.feature.resize(d.feature_count);std::memcpy(r.reward.data(),bytes.data()+layout.reward_offset,4*h.option_count);std::memcpy(r.duration.data(),bytes.data()+layout.duration_offset,2*h.option_count);std::memcpy(r.next_state.data(),bytes.data()+layout.next_offset,8*h.option_count);std::memcpy(r.feature.data(),bytes.data()+layout.feature_offset,4*h.feature_count);r.feature[h.feature_count+p.product_id]=1.0f;if(!r.valid_option_mask)throw std::runtime_error("state has no valid option");for(std::size_t a=0;a<h.option_count;++a)if(valid(r,a)&&r.duration[a]==0)throw std::runtime_error("valid SMDP option has zero duration");d.rows.push_back(std::move(r));}return d;}

std::vector<float> Teacher::q_values(const std::vector<float>& feature)const{std::vector<float>q; q.reserve(option.size());for(const auto& forest:option)q.push_back(predict_forest(forest,feature));return q;}

std::uint32_t predict(const OptionTree& tree,const std::vector<float>& f){if(tree.nodes.empty())return 0;int i=0;for(std::size_t g=0;g<=tree.nodes.size();++g){const auto&n=tree.nodes.at(i);if(n.feature<0)return n.option_id;i=f.at(n.feature)<=n.threshold?n.left:n.right;}return 0;}

FqiReport train_fqi_and_distill(const TransitionDataset& data,const FqiConfig& cfg,int depth,std::size_t min_leaf,double alpha){if(data.rows.empty()||data.option_count<2||cfg.iterations<2||!cfg.trees)throw std::runtime_error("invalid/empty FQI configuration");for(const auto& row:data.rows)if(row.feature.size()!=data.feature_count)throw std::runtime_error("FQI feature width mismatch; product one-hot must be explicit");auto part=partition(data,cfg);auto fit=fit_teacher(data,part.train,part.train_components,cfg,cfg.seed);FqiReport report;report.teacher=std::move(fit.teacher);report.convergence=std::move(fit.convergence);report.train_states=part.train.size();report.holdout_states=part.holdout.size();report.split_components=part.components;report.train_bellman_rmse=bellman_rmse(report.teacher,data,part.train);report.holdout_bellman_rmse=bellman_rmse(report.teacher,data,part.holdout);report.holdout_policy_value=policy_value(&report.teacher,data,part.holdout,1);report.holdout_baseline_value=policy_value(&report.teacher,data,part.holdout,0);report.holdout_oracle_value=policy_value(nullptr,data,part.holdout,2);
    DistillBuilder db{data,report.teacher,{},depth,min_leaf,alpha,{}};db.q.resize(data.rows.size());for(std::size_t i=0;i<data.rows.size();++i)db.q[i]=report.teacher.q_values(data.rows[i].feature);db.build(part.train,0);report.distilled=std::move(db.tree);std::size_t agree=0;for(auto r:part.holdout)agree+=predict(report.distilled,data.rows[r].feature)==greedy(report.teacher,data.rows[r]);report.distilled_teacher_agreement=part.holdout.empty()?0:static_cast<double>(agree)/part.holdout.size();
    if(cfg.bootstrap_rounds){std::vector<double>values(cfg.bootstrap_rounds),agreements(cfg.bootstrap_rounds);std::atomic<std::size_t>next{0};auto threads=std::min<std::size_t>(cfg.bootstrap_rounds,cfg.threads?cfg.threads:std::max(1u,std::thread::hardware_concurrency()));std::vector<std::thread>workers;for(std::size_t w=0;w<threads;++w)workers.emplace_back([&]{for(;;){auto b=next.fetch_add(1);if(b>=cfg.bootstrap_rounds)break;std::mt19937_64 rng(mix(cfg.seed^b^0xfeedULL));std::uniform_int_distribution<std::size_t>pick(0,part.train_components.size()-1);std::vector<std::vector<std::size_t>> comps;std::vector<std::size_t>rows;for(std::size_t k=0;k<part.train_components.size();++k){auto c=part.train_components[pick(rng)];rows.insert(rows.end(),c.begin(),c.end());comps.push_back(std::move(c));}auto local=cfg;local.threads=1;local.bootstrap_rounds=0;auto teacher=fit_teacher(data,rows,comps,local,mix(cfg.seed^b)).teacher;values[b]=policy_value(&teacher,data,part.holdout,1);std::size_t same=0;for(auto r:part.holdout)same+=greedy(teacher,data.rows[r])==greedy(report.teacher,data.rows[r]);agreements[b]=part.holdout.empty()?0:static_cast<double>(same)/part.holdout.size();}});for(auto&w:workers)w.join();report.bootstrap_policy_q025=quantile(values,.025);report.bootstrap_policy_median=quantile(values,.5);report.bootstrap_policy_q975=quantile(values,.975);report.bootstrap_action_agreement=std::accumulate(agreements.begin(),agreements.end(),0.0)/agreements.size();}
    return report;}

void write_report(const std::string& path,const FqiReport&r,const FqiConfig&c,const std::vector<OptionSpec>&options,std::size_t total){std::ofstream o(path);if(!o)throw std::runtime_error("cannot write FQI report");o<<std::setprecision(10)<<"{\n\"schema\":\"g001-dynamic-option-fqi-v1\",\n\"status\":\"trainer-interface; requires real simulator transitions before gameplay claims\",\n\"reward\":\"delta(own_money-opponent_money)\",\n\"gamma\":1,\n\"total_states\":"<<total<<",\"train_states\":"<<r.train_states<<",\"holdout_states\":"<<r.holdout_states<<",\"split_components\":"<<r.split_components<<",\n\"train_bellman_rmse\":"<<r.train_bellman_rmse<<",\"holdout_bellman_rmse\":"<<r.holdout_bellman_rmse<<",\n\"holdout_policy_value\":"<<r.holdout_policy_value<<",\"holdout_baseline_value\":"<<r.holdout_baseline_value<<",\"holdout_oracle_value\":"<<r.holdout_oracle_value<<",\n\"distilled_teacher_agreement\":"<<r.distilled_teacher_agreement<<",\n\"bootstrap\":{\"policy_q025\":"<<r.bootstrap_policy_q025<<",\"policy_median\":"<<r.bootstrap_policy_median<<",\"policy_q975\":"<<r.bootstrap_policy_q975<<",\"action_agreement\":"<<r.bootstrap_action_agreement<<"},\n\"convergence\":[";for(std::size_t i=0;i<r.convergence.size();++i){if(i)o<<',';auto&m=r.convergence[i];o<<"{\"iteration\":"<<m.iteration<<",\"bellman_rmse\":"<<m.bellman_rmse<<",\"max_q_change\":"<<m.max_q_change<<'}';}o<<"],\n\"options\":[";for(std::size_t i=0;i<options.size();++i){if(i)o<<',';o<<"{\"id\":"<<options[i].id<<",\"name\":\""<<options[i].name<<"\"}";}o<<"],\n\"distilled_nodes\":"<<r.distilled.nodes.size()<<",\n\"config\":{\"iterations\":"<<c.iterations<<",\"trees\":"<<c.trees<<",\"tree_depth\":"<<c.tree_depth<<",\"min_leaf\":"<<c.min_leaf<<",\"bootstrap_rounds\":"<<c.bootstrap_rounds<<"}\n}\n";}

} // namespace g001::fqi
