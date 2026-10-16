const fs = require("fs");
const { d, P, H1, buildDoc, runs } = require("./lib");
const parte1 = require("./parte1");
let parte2 = null;
try { parte2 = require("./parte2"); } catch (e) { if (e.code !== "MODULE_NOT_FOUND") throw e; }

const OUT = process.argv[2] || require("path").join(__dirname, "..", "Relatorio_MLP_ENEM.docx");
const GITHUB = process.env.GITHUB_URL || "[link do repositório no GitHub]";

const center = (text, size, opts = {}) => new d.Paragraph({
  alignment: d.AlignmentType.CENTER, spacing: { after: opts.after ?? 120, before: opts.before ?? 0 },
  children: runs(text, { size, bold: opts.bold, color: opts.color }),
});
const highlight = (text) => new d.Paragraph({ alignment: d.AlignmentType.CENTER, spacing: { after: 60 },
  children: [new d.TextRun({ text, size: 22, highlight: "yellow" })] });

const capa = [
  center("Trabalho Prático de Deep Learning", 24, { before: 1200, color: "52514E" }),
  center("Frente 3.1 – Multilayer Perceptrons", 22, { color: "52514E", after: 1400 }),
  center("Previsão da nota média no ENEM a partir de informações anteriores à prova", 40, { bold: true, after: 300 }),
  center("Modelagem, hiperparametrização e avaliação experimental de MLPs com os microdados do INEP (2022–2023)", 24, { color: "52514E", after: 1600 }),
  center("Integrantes", 22, { bold: true, after: 120 }),
  center("João Paulo Monteiro", 22, { after: 60 }),
  center("Pedro Felipe De Moraes Carrijo", 22, { after: 60 }),
  center("Rogérios dos Anjos", 22, { after: 60 }),
  center("", 22, { after: 800 }),
  center("**Conjunto de dados:** Microdados do ENEM – INEP · https://dados.gov.br/dados/conjuntos-dados/inep-microdados-do-enem", 18, { after: 80 }),
  center(`**Código:** ${GITHUB}`, 18, { after: 1200 }),
  center("Outubro de 2026", 22),
  new d.Paragraph({ children: [new d.PageBreak()] }),
];

const sumario = [
  new d.Paragraph({ spacing: { after: 200 }, children: [new d.TextRun({ text: "Sumário", bold: true, size: 30 })] }),
  new d.TableOfContents("Sumário", { hyperlink: true, headingStyleRange: "1-2" }),
  new d.Paragraph({ children: [new d.PageBreak()] }),
];

const corpo = [...(parte2 ? parte2.resumo() : []), ...parte1(), ...(parte2 ? parte2.resultados() : [])];
const doc = buildDoc([...capa, ...sumario, ...corpo], "Previsão da nota média no ENEM com MLP");
fs.mkdirSync(require("path").dirname(OUT), { recursive: true });
d.Packer.toBuffer(doc).then((b) => { fs.writeFileSync(OUT, b); console.log("ok", OUT, b.length); });
