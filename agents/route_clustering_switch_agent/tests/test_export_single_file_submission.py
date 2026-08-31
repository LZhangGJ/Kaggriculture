from export_single_file_submission import _force_target


def main() -> None:
    source = {
        "feature_schema": "semantic_route_switch_v1",
        "nodes": [{"selected": {"opening": "OPEN", "checkpoint": 144}}],
    }
    fixed = _force_target(source, "TARGET")
    node = fixed["nodes"][0]["selected"]
    assert fixed["targets"] == ["TARGET"]
    assert (node["opening"], node["checkpoint"]) == ("OPEN", 144)
    assert node["tree"]["classes"] == ["TARGET"]
    print("FIXED_TARGET_EXPORT_TEST_OK")


if __name__ == "__main__":
    main()
