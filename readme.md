```git clone https://github.com/S-M-production/Reinforcement-Learning-Basics```

```cd ./Reinforcement-Learning-Basics```

```git submodule update --init --recursive```

```cd ./pong-engine```

```git submodule update --init --recursive```

```cmake --version```

```cmake -S . -B build```

```mv ./pong_engine.cpython*.so ../(Whatever folder its needed in)/pong_engine.cpython*.so```

```uv sync```

```uv run```