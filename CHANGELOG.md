# Changelog

## [0.1.8](https://github.com/lazzurs/ringmaster/compare/v0.1.7...v0.1.8) (2026-11-26)


### Features

* argument-aware tool policies and spoken authorisation codes ([#22](https://github.com/lazzurs/ringmaster/issues/22)) ([460ed2a](https://github.com/lazzurs/ringmaster/commit/460ed2ad1314494beac084d847a6134c3d3300c6))

## [0.1.7](https://github.com/lazzurs/ringmaster/compare/v0.1.6...v0.1.7) (2026-11-25)


### Features

* optional automatic approval of tool calls by a local Shieldstral classifier ([#20](https://github.com/lazzurs/ringmaster/issues/20)) ([9c7d483](https://github.com/lazzurs/ringmaster/commit/9c7d483d9e587b62983fde46212c583531493e7c))

## [0.1.6](https://github.com/lazzurs/ringmaster/compare/v0.1.5...v0.1.6) (2026-11-25)


### Features

* alert on database and queue problems, and make every task end ([#18](https://github.com/lazzurs/ringmaster/issues/18)) ([6ea42f7](https://github.com/lazzurs/ringmaster/commit/6ea42f73789f8e1e80394d27ac662755cab4f0b3))

## [0.1.5](https://github.com/lazzurs/ringmaster/compare/v0.1.4...v0.1.5) (2026-11-25)


### Bug Fixes

* ensure one stuck job cannot hold up the whole queue ([#16](https://github.com/lazzurs/ringmaster/issues/16)) ([c2923c7](https://github.com/lazzurs/ringmaster/commit/c2923c73b1c4ae3add4bd4bd8854fcb8ae15d4e7))
* mark LLM credentials sensitive and keep secrets out of logs and errors ([#15](https://github.com/lazzurs/ringmaster/issues/15)) ([67858f4](https://github.com/lazzurs/ringmaster/commit/67858f4909303a4f3e63d68a3608b7ff39088383))

## [0.1.4](https://github.com/lazzurs/ringmaster/compare/v0.1.3...v0.1.4) (2026-11-25)


### Documentation

* add AGENTS.md and relicense under GPL-3.0-or-later ([#13](https://github.com/lazzurs/ringmaster/issues/13)) ([6f80f02](https://github.com/lazzurs/ringmaster/commit/6f80f0221ee05cf2a2131adcc574d797b4992511))

## [0.1.3](https://github.com/lazzurs/ringmaster/compare/v0.1.2...v0.1.3) (2026-11-25)


### Features

* support OpenAI, Mistral, OpenRouter and Ollama as LLM providers ([#11](https://github.com/lazzurs/ringmaster/issues/11)) ([25a3b9d](https://github.com/lazzurs/ringmaster/commit/25a3b9d488ee678aa639948690114d3f6b3f5c89))

## [0.1.2](https://github.com/lazzurs/ringmaster/compare/v0.1.1...v0.1.2) (2026-11-25)


### Features

* HTTP Basic auth and custom headers for MCP servers ([#9](https://github.com/lazzurs/ringmaster/issues/9)) ([40ac6e6](https://github.com/lazzurs/ringmaster/commit/40ac6e6c6c213fae635c4262f200cc2335a963c8))

## [0.1.1](https://github.com/lazzurs/ringmaster/compare/v0.1.0...v0.1.1) (2026-11-24)


### Features

* never fail a job or the service because of an MCP server ([#7](https://github.com/lazzurs/ringmaster/issues/7)) ([3038330](https://github.com/lazzurs/ringmaster/commit/3038330682e43cc031e198083cafa5337aaf71f3))

## 0.1.0 (2026-11-23)


### Features

* initial ringmaster service ([#1](https://github.com/lazzurs/ringmaster/issues/1)) ([806f171](https://github.com/lazzurs/ringmaster/commit/806f171f7e7f1d6376a16bcde33e82d5ad398d4d))
