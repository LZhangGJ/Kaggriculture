// Deployment boundary: construct a decision from one legal observation only.
State public_state(py::dict obs) {
  const int seat = getint(obs, "player");
  if (seat < 0 || seat > 1) throw std::runtime_error("invalid player");
  for (const char* k : {"seed", "rng", "opponent_id"})
    if (obs.contains(k)) throw std::runtime_error("hidden actor input");
  py::dict other(obs.attr("copy")());
  py::dict priv; priv["shed"] = py::dict(); priv["seeds"] = py::dict();
  py::list invs;
  auto farms = py::cast<py::list>(obs["farms"]);
  auto rival = py::cast<py::dict>(farms[1-seat]);
  int count = 1 + py::len(rival["hands"]);
  for (int i=0;i<count;i++) invs.append(py::dict());
  priv["inventories"] = invs; other["private"] = priv;
  py::list both; both.append(seat == 0 ? obs : other); both.append(seat == 1 ? obs : other);
  State s(0); loadstate(s, both); return s;
}

class Actor {
  State state;
  Tasks tasks;
  Ledger ledger;
  vector<Choice> options;
  int seat, worker = 0, markets = 0, units;
  bool stopped = false;
public:
  explicit Actor(py::dict obs): state(public_state(obs)), seat(getint(obs,"player")) {
    units = state.farms[seat].positions.size(); ledger.full_actions = true;
    ledger.begin(state,seat,tasks);
  }
  py::tuple encode_actor() {
    py::array_t<float> tile({48,10,10}), glob(128);
    encode(state,seat,tasks,tile.mutable_data(),glob.mutable_data());
    return py::make_tuple(tile,glob);
  }
  py::tuple menu() {
    options.clear();
    if (!stopped && markets < 10) ledger.menu(worker < units ? worker : -1,options);
    py::array_t<float> f({int(options.size()),NF}); py::array_t<int64_t> b(options.size());
    ledger.features(options,worker < units ? worker : -1,f.mutable_data());
    for (size_t i=0;i<options.size();i++) b.mutable_data()[i]=std::max(1,options[i].q);
    return py::make_tuple(f,b);
  }
  void apply(int index,int q) {
    if (index < 0 || index >= int(options.size()) || q < 1 || q > std::max(1,options[index].q))
      throw std::runtime_error("actor selection outside menu");
    auto c=options[index]; c.q=q; ledger.apply(c,worker < units ? worker : -1);
    if(worker < units) worker++; else {markets++; if(c.verb==STOP) stopped=true;}
    options.clear();
  }
  py::dict action() {
    if(worker < units || (!stopped && markets < 10)) throw std::runtime_error("incomplete actor decision");
    return turndict(ledger.action);
  }
};
