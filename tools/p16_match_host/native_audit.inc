// Audit projections are discarded. Only the existing compiled step advances play.
py::dict advance(Simulator& s, py::sequence raw, int seat) {
  std::array<PlayerAction,2> acts{parse_player(raw[0]),parse_player(raw[1])};
  const auto& units=acts[seat].units;
  py::list completed, invalid;
  int moves=0, idle=0, pickups=0, drops=0, production=0, daytime_overflow=0;
  int hires_before=s.farms()[seat].hires_today;
  auto before=s.project_unit_phase(seat,units,0);
  for(size_t u=0;u<std::min(units.size(),s.farms()[seat].hands.size()+1);u++) {
    const auto& a=units[u];
    auto after=s.project_unit_phase(seat,units,u+1);
    auto pos=u?before.farms()[seat].hands[u-1]:before.farms()[seat].farmer;
    int cell=pos.x+10*pos.y;
    if(a.op>=Op::NORTH&&a.op<=Op::WEST) moves++;
    else if(a.op==Op::PASS) idle++;
    else {
      if(a.op==Op::PICKUP) pickups++;
      else if(a.op==Op::DROP||(a.op==Op::PLACE&&a.item==Item::FERTILIZER)) drops++;
      else production++;
      const auto& b=before.privates()[seat];const auto& n=after.privates()[seat];
      bool changed=!tile(before.farms()[seat].tiles[cell]).equal(tile(after.farms()[seat].tiles[cell]))
          || b.inventories[u]!=n.inventories[u] || b.shed!=n.shed || b.seeds!=n.seeds;
      if(changed) completed.append(py::make_tuple(cell,int(a.op),int(a.item),a.quantity));
      else invalid.append(py::make_tuple(u,pos.x,pos.y));
      if(a.op==Op::DROP) for(int i=0;i<12;i++)
        daytime_overflow+=std::max(0,b.inventories[u][i]+b.shed[i]-n.inventories[u][i]-n.shed[i]);
    }
    before=std::move(after);
  }
  s.step(acts);
  int hires=0,wages=0;
  for(size_t o=0;o<acts[seat].market.size();o++)
    if(acts[seat].market[o].op==Op::HIRE&&s.last_market_fills()[seat][o]) {
      int a=1,b=1;for(int k=0;k<hires_before+hires;k++){int c=a+b;a=b;b=c;}
      wages+=s.config().farm_hand_cost_mult*a;hires++;
    }
  py::dict d;d["completed"]=completed;d["invalid"]=invalid;d["moves"]=moves;d["idle"]=idle;
  d["pickups"]=pickups;d["drops"]=drops;d["production"]=production;
  d["hires"]=hires;d["wages"]=wages;d["overflow_units"]=s.last_end_of_day_overflow()[seat];
  d["daytime_overflow_units"]=daytime_overflow;d["market_fills"]=s.last_market_fills()[seat];
  d["market_cash_shortfalls"]=s.last_market_cash_shortfalls()[seat];return d;
}
PYBIND11_MODULE(_pool_sim,m) {
  py::class_<Simulator>(m,"FastEnv",py::module_local())
    .def(py::init([](uint64_t seed){return Simulator(Config{},seed);}))
    .def("observation",&observation)
    .def("advance",&advance)
    .def("step",[](Simulator&s,py::sequence a){s.step({parse_player(a[0]),parse_player(a[1])});})
    .def_property_readonly("done",&Simulator::done)
    .def_property_readonly("step_count",&Simulator::step_count);
}
