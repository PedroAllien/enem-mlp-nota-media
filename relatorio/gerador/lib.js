// Helpers para montar o relatório com docx-js.
const fs = require("fs");
const path = require("path");
const d = require("docx");

const FONT = "Arial";
const PAGE_W = 11906, PAGE_H = 16838;           // A4
const MARGIN = { top: 1418, bottom: 1134, left: 1418, right: 1134 }; // 2,5/2/2,5/2 cm
const TEXT_W = PAGE_W - MARGIN.left - MARGIN.right; // largura útil em DXA
const ACCENT = "2A78D6", HEADER_FILL = "E8EEF6", BORDER = "C3C2B7";

let figN = 0, tabN = 0;

// texto com **negrito** e *itálico* e `código`
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new d.TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new d.TextRun({ text: t.slice(2, -2), bold: true, ...base }));
    else if (t.startsWith("`")) out.push(new d.TextRun({ text: t.slice(1, -1), font: "Consolas", size: (base.size || 22) - 2, ...base, }));
    else out.push(new d.TextRun({ text: t.slice(1, -1), italics: true, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new d.TextRun({ text: text.slice(last), ...base }));
  return out;
}

const P = (text, opts = {}) => new d.Paragraph({
  children: runs(text, opts.run || {}),
  alignment: opts.align || d.AlignmentType.JUSTIFIED,
  spacing: { after: 120, line: 300, ...(opts.spacing || {}) },
  ...(opts.extra || {}),
});

const H1 = (t) => new d.Paragraph({ heading: d.HeadingLevel.HEADING_1, children: [new d.TextRun(t)], pageBreakBefore: false });
const H2 = (t) => new d.Paragraph({ heading: d.HeadingLevel.HEADING_2, children: [new d.TextRun(t)] });
const H3 = (t) => new d.Paragraph({ heading: d.HeadingLevel.HEADING_3, children: [new d.TextRun(t)] });

const bullets = (items, level = 0) => items.map((t) => new d.Paragraph({
  children: runs(t), numbering: { reference: "bullets", level },
  alignment: d.AlignmentType.JUSTIFIED, spacing: { after: 60, line: 288 },
}));
const numbered = (items, ref = "numbers") => items.map((t) => new d.Paragraph({
  children: runs(t), numbering: { reference: ref, level: 0 },
  alignment: d.AlignmentType.JUSTIFIED, spacing: { after: 60, line: 288 },
}));

function caption(prefix, n, text) {
  return new d.Paragraph({
    alignment: d.AlignmentType.CENTER, spacing: { before: 60, after: 200 }, keepLines: true,
    children: [new d.TextRun({ text: `${prefix} ${n} – `, bold: true, size: 18 }), ...runs(text, { size: 18 })],
  });
}

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}

// figura ocupando `frac` da largura útil
function figure(file, text, frac = 1.0) {
  figN += 1;
  const { w, h } = pngSize(file);
  const widthPx = Math.round((TEXT_W / 1440) * 96 * frac);
  const heightPx = Math.round(widthPx * h / w);
  return [
    new d.Paragraph({ alignment: d.AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 0 },
      children: [new d.ImageRun({ type: "png", data: fs.readFileSync(file), transformation: { width: widthPx, height: heightPx },
        altText: { title: `Figura ${figN}`, description: text.replace(/[*`]/g, ""), name: path.basename(file) } })] }),
    caption("Figura", figN, text),
  ];
}

// tabela: header = [..], rows = [[..]], widths = frações (somam 1), align = ["l","r",...]
function table(text, header, rows, widths, align, opts = {}) {
  tabN += 1;
  const total = TEXT_W;
  const cols = widths.map((f) => Math.round(f * total));
  cols[cols.length - 1] = total - cols.slice(0, -1).reduce((a, b) => a + b, 0);
  const size = opts.size || 17;
  const border = { style: d.BorderStyle.SINGLE, size: 4, color: BORDER };
  const borders = { top: border, bottom: border, left: border, right: border };
  const cell = (txt, i, isHead) => new d.TableCell({
    width: { size: cols[i], type: d.WidthType.DXA }, borders,
    shading: isHead ? { type: d.ShadingType.CLEAR, color: "auto", fill: HEADER_FILL } : undefined,
    margins: { top: 50, bottom: 50, left: 80, right: 80 },
    verticalAlign: d.VerticalAlign.CENTER,
    children: [new d.Paragraph({
      alignment: isHead ? d.AlignmentType.CENTER : ({ l: d.AlignmentType.LEFT, r: d.AlignmentType.RIGHT, c: d.AlignmentType.CENTER }[align[i]] || d.AlignmentType.LEFT),
      spacing: { after: 0, line: 240 },
      children: runs(String(txt), { size, bold: isHead || undefined }),
    })],
  });
  const t = new d.Table({
    width: { size: total, type: d.WidthType.DXA }, columnWidths: cols,
    rows: [
      new d.TableRow({ tableHeader: true, cantSplit: true, children: header.map((h, i) => cell(h, i, true)) }),
      ...rows.map((r) => new d.TableRow({ cantSplit: true, children: r.map((c, i) => cell(c, i, false)) })),
    ],
  });
  const cap = new d.Paragraph({
    alignment: d.AlignmentType.CENTER, spacing: { before: 200, after: 80 }, keepNext: true,
    children: [new d.TextRun({ text: `Tabela ${tabN} – `, bold: true, size: 18 }), ...runs(text, { size: 18 })],
  });
  const out = [cap, t];
  if (opts.note) out.push(new d.Paragraph({ spacing: { before: 40, after: 200 }, children: runs(opts.note, { size: 16, color: "52514E" }) }));
  else out.push(new d.Paragraph({ spacing: { after: 120 }, children: [] }));
  return out;
}

const nextFig = () => figN + 1;
const nextTab = () => tabN + 1;

function buildDoc(children, title) {
  return new d.Document({
    creator: "Grupo", title,
    styles: {
      default: { document: { run: { font: FONT, size: 22 } } },
      paragraphStyles: [
        { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 30, bold: true, font: FONT, color: "0B0B0B" },
          paragraph: { spacing: { before: 360, after: 160 }, outlineLevel: 0, keepNext: true } },
        { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 25, bold: true, font: FONT, color: "184F95" },
          paragraph: { spacing: { before: 260, after: 120 }, outlineLevel: 1, keepNext: true } },
        { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 22, bold: true, font: FONT, color: "52514E" },
          paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 2, keepNext: true } },
      ],
    },
    numbering: { config: [
      { reference: "bullets", levels: [
        { level: 0, format: d.LevelFormat.BULLET, text: "•", alignment: d.AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } },
        { level: 1, format: d.LevelFormat.BULLET, text: "–", alignment: d.AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 270 } } } } ] },
      { reference: "numbers", levels: [
        { level: 0, format: d.LevelFormat.DECIMAL, text: "%1.", alignment: d.AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 300 } } } } ] },
    ] },
    sections: [{
      properties: { page: { size: { width: PAGE_W, height: PAGE_H }, margin: MARGIN } },
      footers: { default: new d.Footer({ children: [new d.Paragraph({ alignment: d.AlignmentType.RIGHT,
        children: [new d.TextRun({ children: [d.PageNumber.CURRENT], size: 18, color: "898781" })] })] }) },
      children,
    }],
  });
}

module.exports = { d, P, H1, H2, H3, bullets, numbered, figure, table, buildDoc, runs, nextFig, nextTab, TEXT_W, ACCENT };
