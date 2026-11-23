---
name: visualization
description: Show customers a chart or diagram when it makes information, comparisons, trends or procedures easier to understand.
  Consider it proactively for decisions and product comparisons, even when a visual was not requested. Also use for edits to recent visuals.
metadata:
  version: 1.0.0
  a-bank-category: PRESENTATION
  a-bank-display-name: Visualization
---

# Visualization

1. Choose a visual when it improves the customer's answer. For example, a grounded comparison of savings terms can help someone weigh a deposit decision. Use the customer's context and evidence to choose the form, and skip a visual when it adds little. Verify current rates or observed trends against relevant sources. Identify hypothetical examples as such.
   Choose the structure, engine, titles, labels, data and settings. Call `present_visualization` with `artifact` matching the supplied schema. After success, answer the customer in `final.answer` with useful context and any required citations. The approved visual appears inline alongside the text.
   Include `layout`, a UI tree you design for the request. Compose `column`, `row` and `grid` (1–4 columns) containers, `heading` and `text` leaves, and exactly one `chart` or `diagram` leaf. An ECharts UI can also place an optional `table` block anywhere. Use `tone` (`plain`, `card`, `accent`) as needed. For example `{"type":"column","children":[{"type":"heading","text":"Your heading"},{"type":"chart"},{"type":"table","text":"Data"}]}`. The app follows your order and nesting.
2. Use `bar` for category comparisons, `stacked_bar` for additive components with the same unit, `line` for ordered observations, and `area` for trends where magnitude matters. Use `pie` or `donut` for distinct nonnegative parts of a whole, up to 12 categories. Use `scatter` for paired numeric observations. Preserve ordering, units, denominators and the distinction between official and simulated values.
3. For ECharts, write the supplied or verified data directly as JSON. Use `x_name`, `x` and `series` with a `name` and `values` for each series. Use strings for category labels and numbers for scatter coordinates and all series values. Choose the palette, axis label, legend, point markers and other allowed settings in `config`. Keep units explicit. Omit the chart if required values are missing. Use numeric scatter coordinates to preserve uneven intervals where appropriate.
4. For Mermaid, choose `flowchart` for procedures, `sequence` or `class` for UML, `er` for entity relationships, `mindmap` for hierarchies, and `architecture` for systems. Supply concise Mermaid source without Markdown fences. Start with `flowchart TD` (or `LR`), `sequenceDiagram`, `classDiagram`, `erDiagram`, `mindmap`, or `architecture-beta` as appropriate. Use at most 48 lines, one statement per line. Use built-in architecture icons such as `server`, `database`, `disk`, `cloud`, or `internet`. Use plain labels, with no styles, links, HTML, callbacks, icon packs or configuration. Describe the diagram's factual steps and relationships in the answer, and cite the supporting procedure or specification for bank facts. Mark proposed or illustrative designs as such.
5. For an ECharts presentation-only edit, use `engine: echarts`, the exact `update_of` ID from `recent_visualizations` and `presentation` with only the changed fields, such as `{"kind":"line"}`. This keeps the dataset and layout. For a data change or Mermaid edit, supply the complete revised artifact with `update_of`. Correct rejected arguments before finishing. Clarify ambiguous targets when necessary. Present only supported and appropriate data.
