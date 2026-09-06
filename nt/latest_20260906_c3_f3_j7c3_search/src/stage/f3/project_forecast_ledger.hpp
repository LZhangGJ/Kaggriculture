#pragma once
// Daily investment forecast ownership, not the actual inventory/market ledger.
// Included after Flow, NT, ND, NP and addflow have been defined.
struct ProjectForecastLedger {
    struct Entry { int pos,item; Flow flow; };
    int day=-1;
    Flow committed{}; // Live assets and their existing continuation forecasts.
    std::vector<Entry> entries; // Exact streams accepted by the base planner.

    static bool same(const Flow&a,const Flow&b,double eps=1e-7){
        for(int d=0;d<ND;d++)for(int i=0;i<NP;i++)
            if(!std::isfinite(a[d][i])||!std::isfinite(b[d][i])||
               std::abs(a[d][i]-b[d][i])>eps)return false;
        return true;
    }
    const Entry* find(int pos)const{
        for(const auto&e:entries)if(e.pos==pos)return &e;
        return nullptr;
    }
    Flow total()const{
        Flow result=committed;
        for(const auto&e:entries)addflow(result,e.flow);
        return result;
    }
    void capture(int current_day,const Flow&background,
                 const std::vector<std::pair<int,int>>&projects,
                 const std::array<Flow,NT>&streams,const Flow&expected){
        ProjectForecastLedger fresh;fresh.day=current_day;fresh.committed=background;
        for(auto[pos,item]:projects){
            if(pos<0||pos>=NT||item<0||item>=NI||fresh.find(pos))
                throw std::runtime_error("forecast ledger capture identity");
            fresh.entries.push_back({pos,item,streams[pos]});
        }
        if(!same(fresh.total(),expected))
            throw std::runtime_error("forecast ledger does not reconstruct accepted plan");
        *this=std::move(fresh);
    }
    Flow replace(int current_day,int pos,int expected_old,int next,const Flow&next_flow){
        if(current_day!=day||pos<0||pos>=NT||next<-1||next>=NI)
            throw std::runtime_error("forecast ledger replacement scope");
        auto it=std::find_if(entries.begin(),entries.end(),[&](const Entry&e){return e.pos==pos;});
        int actual=it==entries.end()?-1:it->item;
        if(actual!=expected_old)throw std::runtime_error("forecast ledger old project mismatch");
        if(!same(next_flow,next_flow)||(next<0&&!same(next_flow,Flow{})))
            throw std::runtime_error("forecast ledger replacement stream");
        if(it!=entries.end()){
            if(next<0)entries.erase(it);
            else *it={pos,next,next_flow};
        }else if(next>=0)entries.push_back({pos,next,next_flow});
        return total();
    }
};
