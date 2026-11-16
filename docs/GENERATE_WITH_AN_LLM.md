# Contribute without coding: generate a component with an LLM

You do not need to know Python or Manim. If you can describe an animation in words, an LLM can write it, and we can turn
it into a contribution. Plan on about 15 minutes.

## 1. Get the code

1. Open [LLM_PROMPT.md](LLM_PROMPT.md) and copy everything below the line into [Claude](https://claude.ai), ChatGPT, Gemini
   or any LLM.
2. Replace the `[BRACKETS]` with what you want to animate and who it is for. Be specific about what should *change* on
   screen: "a pendulum whose length I can set, with the period displayed" beats "something about pendulums".
3. The LLM replies with one Python file (and maybe a test file). That is the contribution.

Using **Claude Code** or Claude with this repository's skill? Just ask: *"Write me a Manimater component that shows ..."*.
The [skill](../skills/manimater-components/SKILL.md) already knows the rules.

## 2. Look at it (optional, but it catches silly mistakes)

If you can run Python, save the code as `manimate/components/community/<name>.py` in a clone of this repo and run
`python -m manimate.preview <name>`; you get a video. If you cannot, skip this step: CI renders it for you.

Even then, read the LLM's summary at the end of its reply and check it says what you wanted. If it is wrong, tell the LLM
what is wrong in plain words ("the labels overlap", "it should start from zero") and ask for a fixed version.

## 3. Submit it

Open the **[New component (from an LLM)](../../../issues/new?template=new-component.yml)** form and fill in:

- the **name** the LLM chose (for example `binary_addition`),
- what it **teaches** in one sentence,
- the **Python code**, pasted in whole,
- the **test code**, if the LLM wrote one.

Then press **Submit new issue**. That is all.

## 4. What happens next

1. A bot reads your submission. It **never runs your code**: it only checks it for anything unsafe (for example network
   or file access) and for the right shape. If something is off, it comments with exactly what to fix, and you can edit
   your issue (or paste the comment back into the LLM) and it will check again.
2. If it passes, the bot opens a **pull request** with your file, credited to you.
3. A maintainer reads the code, runs it, renders the preview video, and may ask for changes. You can answer in the
   pull request in plain words; the LLM can help you reply.
4. When it is merged, it becomes part of the library, and it flows into [Manimater](https://github.com/bishnt/manimater)
   automatically.

## Tips for a component that gets merged

- **One idea.** "How a transformer works" is a whole video; "attention weights between two words" is a component.
- **Reusable.** Parameters (numbers, labels, lists) should make it useful for many examples, not just yours.
- **Plain on-screen text**, short labels, nothing that overflows the screen.
- **Honest content.** Check that the maths or the facts are correct. The reviewer will, and so should you.

## Licence

By submitting you agree your contribution is released under the [MIT licence](../LICENSE), and that you have the right to
share it (you made it, or an LLM made it for you and you are not copying someone else's code).
