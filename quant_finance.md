---
layout: default
title: Quantitative Finance
permalink: /quant_finance
---

# ToC
- [QuantLib on a Mac](#quantlib-on-a-mac)

<hr>

# QuantLib on a Mac
Installing and using `QuantLib` on a mac is timeless entertainment. Although I've done it several times in the past, both on mac os x and on windows machines, it always takes me a few hours of googling and troubleshooting, and it always leaves me with the same feeling that it was all avoidable frustration since the online instructions were clear. I don't know why this is the case because I have a natural proclivity for attention to detail, but always somethings falls between the cracks. So I decided to document the process for my own benefit in a way that highlights the little things that derail the process and which, in hindsight, are obvious.

- Get the library

  QuantLib can be downloaded from here: [QuantLib](https://github.com/lballabio/QuantLib/releases)

  On a mac, you'd want the `tar` file.

- Follow the installation instructions: [Installation](https://www.quantlib.org/install/macosx.shtml)

  - **Boost:** The first step in the installation guide pertains to the `boost` C++ library (libraries?) which is dependency of QuantLib's. Using `Homebrew` to install `boost` works perfectly fine (for my purposes, at least &mdash; I haven't had to look back).

  - **Compile from source:** For QuantLib itself, however, I want the source code and the ability to modify it so I go directly to the `configure` step of the guide.

  
