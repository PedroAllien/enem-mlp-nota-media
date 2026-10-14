const fs = require("fs");
const readCsv = require("./csv");
const { H1, P } = require("./lib");
const TAB = (f) => `${require("path").join(__dirname, "..", "..", "outputs")}/tables/${f}`;
const has = (f) => fs.existsSync(TAB(f));
const opt = (m) => { try { return require(m); } catch (e) { if (e.code === "MODULE_NOT_FOUND") return null; throw e; } };

module.exports = {
  resumo() { const m = opt("./sec_resumo"); return m ? m() : []; },
  resultados() {
    const out = [H1("4. Resultados de validação e seleção dos modelos")];
    out.push(...require("./sec_capacidade")(readCsv(TAB("capacidade.csv"))));
    out.push(...require("./sec_regularizacao")(readCsv(TAB("regularizacao.csv")), readCsv(TAB("diagnostico_l2.csv"))));
    out.push(...require("./sec_optuna_espaco")());
    const r = opt("./sec_optuna_resultados"); if (r) out.push(...r());
    const f = opt("./sec_finais"); if (f) out.push(...f());
    const l = opt("./sec_conclusao"); if (l) out.push(...l());
    return out;
  },
};
