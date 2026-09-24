构建：`python3 build.py --unit`。需要 C++20 编译器。

编译旗标在 COMPILER_FLAGS.json；实际输入哈希和编译器在 policy/a06.BUILD.json。不同编译器的二进制字节不保证相同，应重跑提供的接口/对局验证。
