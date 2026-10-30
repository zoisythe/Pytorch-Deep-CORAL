# Beamer Presentation

**This is a substitute, not the original.** The template historically served
at this URL was Vel's Beamer presentation deck (LaTeXTemplates.com), licensed
CC-BY-NC-SA — non-commercial, so it cannot be redistributed here. This
download is a different, permissively-licensed Beamer template of the same
kind: `klb2/beamer-presentation-template` (Apache-2.0), built on the popular
open-source metropolis theme.

## Files

- `main.tex` — the presentation. Start here.
- `setup-colors.tex` — the four theme colours (accent/background/foreground).
- `setup-beamer.tex` — beamer font/template tweaks (title page layout, a
  `\seprule` helper for a separator line).
- `setup-math.tex` — math operators and theorem environments.
- `setup-plots.tex` — a colour-blind-safe PGFPlots cycle (Paul Tol "High
  Contrast") and a `betterplot` style.
- `example-content.tex` — a full set of demo slides: text, math, theorems,
  typography, columns, tables, blocks, a line plot and a standout slide.
  Delete and replace with your own content.

## Building

    lualatex main

Requires **LuaLaTeX**. The metropolis theme's default fonts are Fira Sans and
Fira Mono; the pinned compiler only resolves them by file (via kpathsea), not
by system family name, so `main.tex` loads the exact `.otf` files by name
(`\setsansfont{FiraSans-Light}[...]`, `\setmonofont{FiraMono-Regular}[...]`)
rather than relying on metropolis's own family-name auto-detection, which
would otherwise silently fall back to a plain default font with only a
build-log warning.

## Customization

Edit the four colours in `setup-colors.tex`. Everything else — title page
layout, math macros, plot styling — lives in its own `setup-*.tex` file so a
change stays easy to find.

## Licence

`beamer-presentation-template` is licensed **Apache-2.0**,
© Karl-Ludwig Besser. It builds on the **metropolis** Beamer theme (CC-BY-SA
4.0, © Matthias Vogelgesang and contributors), which ships with TeX Live and
is not redistributed here. Source:
https://github.com/klb2/beamer-presentation-template

This starter document is a minimal example written for latextemplates.com and
carries no additional restrictions.
