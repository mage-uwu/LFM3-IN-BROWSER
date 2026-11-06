# User guide

**LFM3 in the Browser** runs Liquid AI's LFM3-Omni-2B, a model that understands text, photos and speech and can
answer out loud, entirely on your own device. It is a reference implementation by Mage / Toaster Club LLC, meant to show how to put the model directly inside a website or web app. AI coding agents: see
[AGENTS.md](AGENTS.md).

**Open it:** <https://lfm.toasterclub.com>

## What you need

- **A browser with WebGPU.** That means Safari 26 or later on iPhone or iPad, or a recent Chrome, Edge or Safari
  on a computer.
- **About 1 GB free** for the language model, plus about 0.3 GB more the first time you use photos, voice or spoken
  answers.
- **The first load downloads about 0.9 GB.** After that, the model loads from your device.

## Getting started

1. Open the site and tap **Load the model**. The first time, it downloads and saves a copy in your browser's
   storage on this device. A download that is interrupted continues where it stopped next time.
2. **On iPhone,** in Safari, tap Share → **Add to Home Screen**. It then opens like an app, with its own icon.

## Using it

- **Type** a message and tap the arrow. The answer is written out as it is generated.
- **Talk:** hold the round **mic** button, speak, and let go. Or tap once to start and tap again to send.
  - The answer is spoken aloud and written in the chat. Talking in a new chat turns on voice mode.
  - The soap bubble above the message box shows what it is doing: it listens, thinks, then talks.
  - Press again while it is talking to interrupt it.
- **Live:** tap the **camera** button. The camera fills the screen; hold the bubble and talk about what it sees.
  The picture from when you start talking goes in with your words. Use the buttons at the bottom to turn the
  camera off, switch cameras, or end Live.
- **Photos and documents:** tap **+** to attach a picture or a text file (.txt, .md, .csv, .json, .html and
  similar), then ask about it. A document is cut into passages, and each typed message gets the passages that match
  it best. Documents belong to the chat they were added to: **New chat** clears them. Tap **×** on a document's chip
  to remove it.
- **Spoken answers to typed messages:** the **speaker** button in the top bar.
- **Instructions** (the **sliders**): give it your own system prompt, for example "Always answer in French." This
  starts a new chat.
- **Smoke test** (also in the **sliders** sheet): **Run the smoke test** tries every workflow once on your device
  and shows, for each, whether it worked and how fast:
  - text: time to the first token, then prefill and decode tokens per second;
  - a photo, a spoken reply, and speech recognition;
  - a push-to-talk turn with a camera picture: the time from letting go to the answer's first sound.

  It plays two short spoken answers and starts a new chat when it is done. **Copy results**, **Share** or **Save
  JSON** to keep or post the numbers. Until you do, they stay on the device.
- **The top bar** names the model and shows two tags: your GPU, as the browser names it (for example "apple
  metal-3"), and the weights' precision ("4-bit").
- **New chat:** the **pencil** button.
- **Decide** ([decide.html](https://lfm.toasterclub.com/decide.html), linked from the setup card and About): a
  separate page for Liquid's d1-Omni-2B decision model. Put text or JSON in the box, or add a photo or a recording,
  then write questions: **Yes / no**, **One of**, **Any of** (each option yes or no), or **Score** (levels, lowest
  first). **Decide** answers them all at once, with a bar for each option's probability. **Try one** fills in an
  example. It downloads its own 0.93 GB model the first time (plus 0.19 GB for photos and 0.06 GB for recordings)
  and keeps a copy, separate from the chat's; **Storage and about** removes it.
- **Voice Desk** ([voice.html](https://lfm.toasterclub.com/voice.html)): both models at once. Tap the round button,
  speak, and tap again. d1 decides first (its card shows each answer), then LFM3-Omni-2B answers aloud with that
  decision in mind. Pick a **Profile** (customer line, personal assistant, language tutor) or edit the questions and
  the voice's instructions yourself. The star button tries a sample message spoken by the model's own voice. It
  needs room on the GPU for both models (about 2.1 GB); phones with less may not load it.

## Privacy

Everything runs on your device: what you type, say and show never leaves it. The site sends you the app and the
model's files and receives nothing back. The model's copy, your instructions and the last smoke test's results are
kept in this browser's storage on this device. Conversations and documents are not stored: they last only while
the chat is open. Clearing the site's data in your browser settings removes everything, and **Remove the saved
copy** in Settings → Storage deletes the model's copy.

## If something goes wrong

| What you see | What to do |
| --- | --- |
| "WebGPU is not available" | Update Safari to 26 or later, or use a recent Chrome or Edge. On a computer, check that hardware acceleration is on. |
| The microphone or camera does not start | Allow them for this site in the browser's settings. Live needs both; Talk needs the microphone. |
| You hear nothing | Turn up the volume, and check the iPhone's silent switch. |
| Speech pauses between phrases | Your device makes speech a little slower than it plays, so it waits to play smoothly. It goes faster on newer devices. |
| "Not enough room" while loading | Free some storage. The model still runs, but it downloads again next time. |
| It forgets the start of a long conversation | The model holds about 4,000 tokens of conversation. Past that, it says so and starts over, and it still remembers what it learned about you. For a new topic, start a **New chat**. |

## Put it in your own website or app

- **Link to it.** <https://lfm.toasterclub.com> works on any device with WebGPU.
- **Host your own copy.** Fork the repository and deploy the `public/` folder to Cloudflare Workers or any static
  host; the [README](README.md#deploy-your-own) has the steps. There is no server code to run.
- **Use the engine in your page:**

  ```js
  import { loadLFM3 } from "./engine/loader.js";
  const { chat } = await loadLFM3({ onProgress: (p) => console.log(`${p.got} / ${p.total}`) });
  const { reply } = await chat.send("Hello!", { onText: (t) => console.log(t) });
  ```

  [`public/examples/chat.html`](public/examples/chat.html) is a complete working page, and
  [docs/API.md](docs/API.md) covers photos, speech and spoken answers.

## Terms of use

The model is **LFM3-Omni-2B by Liquid AI**, used under the **LFM Open License v1.0**. Its full text is in
[`LICENSE-LFM.txt`](LICENSE-LFM.txt) and on the site under `licenses/`. In short, and the license text governs:

- You may use, modify and share the model. Anyone you share it with must get the license too. Mark any files you
  change, and keep the attributions and the NOTICE.
- **Commercial use is licensed only for individuals and organizations below US$10 million in annual revenue.**
  Above that, you need an agreement with Liquid AI. Qualified non-profits doing non-commercial or research work are
  exempt from the limit.
- Liquid AI's names may only be used to say where the model comes from. This project is not affiliated with or
  endorsed by Liquid AI.
- The model comes with no warranty, and the license ends if its terms are broken.

The app and engine code are under the [Apache License 2.0](LICENSE), copyright 2026 Toaster Club LLC.
