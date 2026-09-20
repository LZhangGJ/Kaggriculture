#include "route_loader.hpp"

#include <zlib.h>

#include <array>
#include <charconv>
#include <fstream>
#include <regex>
#include <stdexcept>
#include <string_view>
#include <unordered_map>

namespace g001::repair {
namespace {

class JsonCursor {
public:
    explicit JsonCursor(std::string_view text) : text_(text) {}
    void whitespace() {
        while (position_ < text_.size() &&
               (text_[position_] == ' ' || text_[position_] == '\n' ||
                text_[position_] == '\r' || text_[position_] == '\t')) ++position_;
    }
    char peek() { whitespace(); return position_ < text_.size() ? text_[position_] : '\0'; }
    bool take(char value) {
        whitespace();
        if (position_ < text_.size() && text_[position_] == value) {
            ++position_; return true;
        }
        return false;
    }
    void expect(char value) {
        if (!take(value)) throw std::runtime_error("invalid route JSON");
    }
    std::string string() {
        whitespace(); expect('"'); std::string result;
        while (position_ < text_.size()) {
            const char value = text_[position_++];
            if (value == '"') return result;
            if (value != '\\') { result.push_back(value); continue; }
            if (position_ >= text_.size()) break;
            const char escaped = text_[position_++];
            switch (escaped) {
                case '"': case '\\': case '/': result.push_back(escaped); break;
                case 'b': result.push_back('\b'); break;
                case 'f': result.push_back('\f'); break;
                case 'n': result.push_back('\n'); break;
                case 'r': result.push_back('\r'); break;
                case 't': result.push_back('\t'); break;
                default: throw std::runtime_error("unsupported JSON escape");
            }
        }
        throw std::runtime_error("unterminated JSON string");
    }
    int integer() {
        whitespace(); const auto* begin = text_.data() + position_;
        const auto* end = text_.data() + text_.size(); int value = 0;
        const auto parsed = std::from_chars(begin, end, value);
        if (parsed.ec != std::errc{}) throw std::runtime_error("expected JSON integer");
        position_ = static_cast<std::size_t>(parsed.ptr - text_.data()); return value;
    }
    void skip() {
        whitespace(); const auto value = peek();
        if (value == '"') { (void)string(); return; }
        if (value == '{') {
            expect('{'); if (take('}')) return;
            do { (void)string(); expect(':'); skip(); } while (take(','));
            expect('}'); return;
        }
        if (value == '[') {
            expect('['); if (take(']')) return;
            do { skip(); } while (take(',')); expect(']'); return;
        }
        while (position_ < text_.size() && text_[position_] != ',' &&
               text_[position_] != ']' && text_[position_] != '}' &&
               text_[position_] != ' ' && text_[position_] != '\n' &&
               text_[position_] != '\r' && text_[position_] != '\t') ++position_;
    }
private:
    std::string_view text_;
    std::size_t position_ = 0;
};

fastkag::Op parse_op(std::string_view value) {
    static const std::unordered_map<std::string_view, fastkag::Op> values{
        {"PASS",fastkag::Op::PASS},{"NORTH",fastkag::Op::NORTH},
        {"SOUTH",fastkag::Op::SOUTH},{"EAST",fastkag::Op::EAST},
        {"WEST",fastkag::Op::WEST},{"DROP",fastkag::Op::DROP},
        {"PICKUP",fastkag::Op::PICKUP},{"PLACE",fastkag::Op::PLACE},
        {"PLANT",fastkag::Op::PLANT},{"WATER",fastkag::Op::WATER},
        {"HARVEST",fastkag::Op::HARVEST},{"FERTILIZE",fastkag::Op::FERTILIZE},
        {"DIG",fastkag::Op::DIG},{"BUILD_COOP",fastkag::Op::BUILD_COOP},
        {"BUILD_PASTURE",fastkag::Op::BUILD_PASTURE},{"FEED",fastkag::Op::FEED},
        {"COLLECT_FERTILIZER",fastkag::Op::COLLECT_FERTILIZER},
        {"CARE",fastkag::Op::CARE},{"HIRE",fastkag::Op::HIRE},
        {"BUY_LAND",fastkag::Op::BUY_LAND},{"BUY_SEED",fastkag::Op::BUY_SEED},
        {"BUY_PRODUCT",fastkag::Op::BUY_PRODUCT},
        {"BUY_ANIMAL",fastkag::Op::BUY_ANIMAL},{"SELL",fastkag::Op::SELL},
    };
    const auto found = values.find(value);
    if (found == values.end()) throw std::runtime_error("unknown route operation");
    return found->second;
}

fastkag::Item parse_item(std::string_view value) {
    static const std::unordered_map<std::string_view, fastkag::Item> values{
        {"WHEAT",fastkag::Item::WHEAT},{"CARROT",fastkag::Item::CARROT},
        {"TOMATO",fastkag::Item::TOMATO},{"STRAWBERRY",fastkag::Item::STRAWBERRY},
        {"MELON",fastkag::Item::MELON},{"EGG",fastkag::Item::EGG},
        {"MILK",fastkag::Item::MILK},{"WOOL",fastkag::Item::WOOL},
        {"FERTILIZER",fastkag::Item::FERTILIZER},{"GOOSE",fastkag::Item::GOOSE},
        {"COW",fastkag::Item::COW},{"SHEEP",fastkag::Item::SHEEP},
    };
    const auto found = values.find(value);
    if (found == values.end()) throw std::runtime_error("unknown route item");
    return found->second;
}

fastkag::Action parse_action(JsonCursor& json) {
    json.expect('['); fastkag::Action result; result.op = parse_op(json.string());
    if (json.take(',')) {
        result.item = parse_item(json.string());
        if (json.take(',')) result.quantity = json.integer();
    }
    while (json.take(',')) json.skip();
    json.expect(']');
    return result;
}

std::vector<fastkag::Action> parse_actions(JsonCursor& json) {
    std::vector<fastkag::Action> result; json.expect('[');
    if (json.take(']')) return result;
    do { result.push_back(parse_action(json)); } while (json.take(','));
    json.expect(']'); return result;
}

fastkag::PlayerAction parse_turn(JsonCursor& json) {
    fastkag::PlayerAction result; fastkag::Action farmer; bool has_farmer = false;
    json.expect('{');
    if (!json.take('}')) {
        do {
            const auto key = json.string(); json.expect(':');
            if (key == "farmer") { farmer = parse_action(json); has_farmer = true; }
            else if (key == "hands") result.units = parse_actions(json);
            else if (key == "market") result.market = parse_actions(json);
            else json.skip();
        } while (json.take(','));
        json.expect('}');
    }
    result.units.insert(result.units.begin(), has_farmer ? farmer : fastkag::Action{});
    return result;
}

std::vector<fastkag::PlayerAction> parse_tape(JsonCursor& json) {
    std::vector<fastkag::PlayerAction> result; json.expect('[');
    if (json.take(']')) return result;
    do { result.push_back(parse_turn(json)); } while (json.take(','));
    json.expect(']'); return result;
}

std::string inflate_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open compressed route tapes: " + path);
    std::vector<unsigned char> compressed(
        (std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    z_stream stream{}; stream.next_in = compressed.data();
    stream.avail_in = static_cast<uInt>(compressed.size());
    if (inflateInit(&stream) != Z_OK) throw std::runtime_error("zlib init failed");
    std::string output; std::array<char, 1 << 16> buffer{}; int status = Z_OK;
    while (status == Z_OK) {
        stream.next_out = reinterpret_cast<Bytef*>(buffer.data());
        stream.avail_out = static_cast<uInt>(buffer.size());
        status = inflate(&stream, Z_NO_FLUSH);
        output.append(buffer.data(), buffer.size() - stream.avail_out);
    }
    inflateEnd(&stream);
    if (status != Z_STREAM_END) throw std::runtime_error("route decompression failed");
    return output;
}

std::string read_text(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open route library: " + path);
    return {(std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>()};
}

std::string resolve_route(const std::string& path, const std::string& requested) {
    if (requested.size() < 2 || requested.front() != 'G') return requested;
    const auto source = read_text(path);
    const std::regex pattern("\\\"family\\\"\\s*:\\s*\\\"" + requested +
        "\\\"[^}]*\\\"route_id\\\"\\s*:\\s*\\\"([^\\\"]+)\\\"");
    std::smatch match;
    if (!std::regex_search(source, match, pattern))
        throw std::runtime_error("route family not found: " + requested);
    return match[1].str();
}

}  // namespace

std::vector<fastkag::PlayerAction> load_route(const std::string& compressed_tapes_path,
                                               const std::string& route_library_path,
                                               const std::string& family_or_route_id) {
    const auto route_id = resolve_route(route_library_path, family_or_route_id);
    const auto payload = inflate_file(compressed_tapes_path);
    JsonCursor json(payload); json.expect('{');
    if (!json.take('}')) {
        do {
            const auto key = json.string(); json.expect(':');
            if (key == route_id) return parse_tape(json);
            json.skip();
        } while (json.take(','));
        json.expect('}');
    }
    throw std::runtime_error("route id not found: " + route_id);
}

}  // namespace g001::repair
