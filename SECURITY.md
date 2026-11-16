# Security

Components are pure drawing code and are run by services that render videos for other people, so security matters here.

- Components must not read files, use the network, spawn processes, or use `eval`/`exec`. CI enforces the obvious cases
  for everything in `manimate/components/community/`, and maintainers review every pull request.
- Parameters are validated against the catalog before a component runs; keep validators strict.

To report a vulnerability, please do **not** open a public issue. Use GitHub's
[private vulnerability reporting](https://github.com/bishnt/manimater-components/security/advisories/new) for this repository.
