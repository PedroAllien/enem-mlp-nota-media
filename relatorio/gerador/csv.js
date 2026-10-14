const fs = require("fs");
// CSV simples (sem vírgulas dentro de campos entre aspas, exceto listas entre colchetes)
module.exports = function readCsv(file) {
  const lines = fs.readFileSync(file, "utf8").trim().split(/\r?\n/);
  const split = (l) => { const out = []; let cur = "", q = false, br = 0;
    for (const ch of l) { if (ch === '"') { q = !q; continue; } if (ch === "[") br++; if (ch === "]") br--;
      if (ch === "," && !q && br === 0) { out.push(cur); cur = ""; } else cur += ch; }
    out.push(cur); return out; };
  const head = split(lines[0]);
  return lines.slice(1).map((l) => { const v = split(l); const o = {}; head.forEach((h, i) => (o[h] = v[i])); return o; });
};
