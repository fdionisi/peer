<div align="center">
  <h1>Peer</h1>
  <p>
    <b>
      An open, (eventually) voice-first peer for your computer.
    </b>
  </p>
</div>

## Abstract

This is a personal exploration, not a product. It is only a few days old and
much of it was built with AI; at this point, it's the ideas I'm
trying to feel out by building them.

At the centre sits the `brain`: one continuous conversation behind a plain
interface. It compresses large context on its own, detects when a topic drifts
and splits the thread for you, and recalls what's relevant when you need it. As
an avid AI user, my pain is managing threads by hand, and I wanted something
that just did it for me, spoken to in plain language. Nothing too new, really.

But the `brain` is _just_ the brain. **Peer** is the whole idea: the brain is
the cognition, and it may live anywhere, while a native client owns the parts that touch you and your
computer. I keep noticing how much tooling in this space is platform-specific,
wiring up bespoke integrations app by app. My instinct runs the other way: lean
on the OS accessibility APIs and integrate computer-use natively, so Peer
operates your machine the way you do rather than through one-off connectors.

What I'm really after is a smooth, continuous speech-to-speech experience,
because I increasingly suspect the chat window is the wrong abstraction. I want
to explore language models as a computer's input surface, adjacent to the mouse
and keyboard, giving you a different, additional way to
interact with your device.

## Getting started

You'll need [Rust](https://rustup.rs), [Docker](https://docs.docker.com/get-docker/),
a [Mistral](https://console.mistral.ai) API key, and optionally an
[Exa](https://exa.ai) key for web access.

```sh
# 1. Start SurrealDB
docker compose up -d

# 2. Point the brain at your keys
export MISTRAL_API_KEY=...
export EXA_API_KEY=...        # optional, enables web search

# 3. Run the TUI
cargo run -p brain_cli
```

The SurrealDB defaults (`root` / `root` at `ws://127.0.0.1:8000`) match the
bundled `docker-compose.yml`, so nothing else needs configuring to run locally.

## What's here

- **Language model access**, streaming text and tool calls.
- **Automatic compression**, detecting topic drift and splitting threads on its own.
- **Storage**, persisting conversations and recalls.
- **Tools**, with basic Exa giving web access.

## What's next

In rough order of what I want to reach for:

- [ ] **Accessibility-based computer-use** by integrating [`agent-desktop`](https://github.com/lahfir/agent-desktop) , so Peer can observe and operate any application through the OS accessibility tree rather than bespoke per-app connectors.
- [ ] **STT input interface** for transcription-at-the-edge, feeding plain text into the brain, potentially behind a trait..
- [ ] **Stronger evals** given there's exactly one in the can today, and not very strong (topic-drift detection).

## License

_Peer_ is distributed under the terms of the MIT license.

See [LICENSE](LICENSE) for details.
