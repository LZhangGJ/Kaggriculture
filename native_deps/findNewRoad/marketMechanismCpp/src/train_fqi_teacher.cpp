#include "fqi_teacher.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

int main(int argc,char**argv){
    try{
        std::string input,output;g001::fqi::FqiConfig cfg;int distill_depth=4;std::size_t distill_leaf=512;double distill_alpha=.001;
        for(int i=1;i<argc;++i){std::string_view arg=argv[i];auto value=[&](){if(++i>=argc)throw std::runtime_error("missing value for "+std::string(arg));return std::string(argv[i]);};
            if(arg=="--input")input=value();else if(arg=="--output")output=value();else if(arg=="--iterations")cfg.iterations=std::stoull(value());else if(arg=="--trees")cfg.trees=std::stoull(value());else if(arg=="--tree-depth")cfg.tree_depth=std::stoi(value());else if(arg=="--min-leaf")cfg.min_leaf=std::stoull(value());else if(arg=="--random-features")cfg.random_features=std::stoull(value());else if(arg=="--random-thresholds")cfg.random_thresholds=std::stoull(value());else if(arg=="--holdout")cfg.holdout_fraction=std::stod(value());else if(arg=="--seed")cfg.seed=std::stoull(value());else if(arg=="--threads")cfg.threads=std::stoull(value());else if(arg=="--bootstrap")cfg.bootstrap_rounds=std::stoull(value());else if(arg=="--distill-depth")distill_depth=std::stoi(value());else if(arg=="--distill-min-leaf")distill_leaf=std::stoull(value());else if(arg=="--distill-alpha")distill_alpha=std::stod(value());else if(arg=="--help"){std::cout<<"train_fqi_teacher --input transitions.mfqi --output report.json [--threads N --iterations N --trees N --bootstrap N]\n";return 0;}else throw std::runtime_error("unknown argument: "+std::string(arg));}
        if(input.empty()||output.empty())throw std::runtime_error("--input and --output required");
        auto data=g001::fqi::load_transitions(input);auto report=g001::fqi::train_fqi_and_distill(data,cfg,distill_depth,distill_leaf,distill_alpha);
        std::vector<g001::fqi::OptionSpec> options;for(std::size_t i=0;i<data.option_count;++i)options.push_back({static_cast<std::uint32_t>(i),"option-"+std::to_string(i)});
        auto parent=std::filesystem::path(output).parent_path();if(!parent.empty())std::filesystem::create_directories(parent);g001::fqi::write_report(output,report,cfg,options,data.rows.size());
        std::cout<<"states="<<data.rows.size()<<" options="<<data.option_count<<" train="<<report.train_states<<" holdout="<<report.holdout_states<<" bellman_holdout="<<report.holdout_bellman_rmse<<" policy="<<report.holdout_policy_value<<" baseline="<<report.holdout_baseline_value<<" oracle="<<report.holdout_oracle_value<<" distilled_agreement="<<report.distilled_teacher_agreement<<" report="<<output<<'\n';return 0;
    }catch(const std::exception&e){std::cerr<<"fatal: "<<e.what()<<'\n';return 1;}
}
