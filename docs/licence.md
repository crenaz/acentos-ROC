# Licence

Copyright (c) 2026 crenaz. Licensed under **AGPL-3.0-only**.

The full text is in [`LICENSE`](../LICENSE), `pyproject.toml` declares it as an SPDX
expression — so the built wheel carries `License-Expression: AGPL-3.0-only` and
bundles the text — and every source file opens with an SPDX header.

The AGPL is the GPL plus one extra obligation, [section 13](../LICENSE): if you run a
modified version and let other people use it **over a network**, those users are
entitled to your modified source. Merely running it privately, or on your own
documents, triggers nothing.

In practice:

| What you do | What you owe |
| --- | --- |
| Run it on your own images | nothing |
| Modify it and keep the changes to yourself | nothing |
| Distribute it, modified or not | source, under AGPL-3.0 |
| Offer it as a hosted service, modified | source, under AGPL-3.0, to your users |

A commercial licence is available on request for anyone who needs to build on this
without those obligations.

Contributions are accepted under AGPL-3.0 **plus** a grant allowing the copyright
holder to relicense them, which is what keeps that commercial option open. See
[`CONTRIBUTING.md`](../CONTRIBUTING.md).

> Dependencies keep their own terms and none of them is AGPL: OpenCV is Apache-2.0,
> NumPy, pandas and Pillow are BSD-style, pytesseract is Apache-2.0, and pypdfium2
> wraps PDFium under BSD-3-Clause/Apache-2.0. Tesseract itself is Apache-2.0 and is
> invoked as a separate process, not linked. The `tessdata_best` models fetched by
> `scripts/fetch_tessdata.sh` are Apache-2.0 and are gitignored rather than
> redistributed here.
