// Seção 4.3 (resultados) e 4.4 (seleção)
const fs = require("fs");
const { P, H2, H3, bullets, figure, table, nextFig, nextTab } = require("./lib");
const readCsv = require("./csv");
const FIG = (f) => `${require("path").join(__dirname, "..", "..", "outputs")}/figures/${f}.png`;
const TAB = (f) => `${require("path").join(__dirname, "..", "..", "outputs")}/tables/${f}`;
const n = (x, k = 2) => Number(x).toLocaleString("pt-BR", { minimumFractionDigits: k, maximumFractionDigits: k });
const int = (x) => Math.round(Number(x)).toLocaleString("pt-BR");
const sci = (x) => { const v = Number(x); const e = Math.floor(Math.log10(v)); const m = v / 10 ** e;
  const sup = String(e).replace("-", "⁻").replace(/\d/g, (d) => "⁰¹²³⁴⁵⁶⁷⁸⁹"[d]); return `${n(m, 1)}·10${sup}`; };

module.exports = function optunaResultados() {
  const trials = readCsv(TAB("optuna_trials.csv"));
  const comp = trials.filter((t) => t.state === "COMPLETE").sort((a, b) => a.value - b.value);
  const sel = JSON.parse(fs.readFileSync(TAB("selecao_arquiteturas.json"), "utf8"));
  const out = [];
  out.push(H3("Resultados da busca"));
  out.push(P(`O estudo rodou 60 trials em 25 minutos de GPU: **36 completos e 24 podados**. Quase todas as podas (19 de 24) ocorreram na 4ª época, o primeiro ponto em que o pruner pode agir. Um trial podado custou em média 11,5 s, contra 33,6 s de um completo, o que dá uma economia estimada de ~9 minutos, cerca de um quarto do tempo total que a busca teria sem poda. O histórico (Figura ${nextFig()}) mostra convergência rápida. Os 12 trials aleatórios iniciais chegaram a 69,00; a fase guiada pelo TPE encontrou 68,89 no trial 13 e o melhor valor, **68,84, no trial 33**. Nenhum dos 26 trials seguintes superou esse valor (o mais próximo foi 68,85), sinal de que o orçamento de 60 foi suficiente. O ganho total da busca sobre a configuração escolhida à mão (trial 0: 69,18) foi de 0,33 ponto.`));
  out.push(...figure(FIG("optuna_historico"), "Histórico da otimização: RMSE de validação por trial (cinza = valor no momento da poda) e melhor valor acumulado."));
  out.push(P(`**O que importou.** A importância por fANOVA (Figura ${nextFig()}) mede quanto da variação do RMSE entre trials cada hiperparâmetro explica. Os de **regularização** (L2: 0,23; dropout: 0,21) e os de **representação e otimização** (embedding do município e normalização: 0,14 cada; taxa de aprendizado: 0,12) dominam. **Profundidade (0,02), largura (0,03) e formato (0,00) quase não importam**, o que confirma de forma independente o resultado de E1. Ativação (0,06) e tamanho do lote (0,05) ficaram no meio.`));
  out.push(...figure(FIG("optuna_importancia"), "Importância dos hiperparâmetros (fANOVA) sobre o RMSE de validação dos trials completos.", 0.75));
  out.push(P(`A Figura ${nextFig()} mostra o RMSE de cada trial completo contra cada hiperparâmetro:`));
  out.push(...bullets([
    "**Região ótima bem definida:** os 10 melhores trials (68,84 a 68,89 — diferença de 0,05 ponto) **todos** usam LayerNorm, SiLU, dropout entre 0,34 e 0,49, lote 1024 e embedding de dimensão 16, com L2 entre 1,6·10⁻⁶ e 4,9·10⁻⁵ e taxa de aprendizado entre 4,5·10⁻⁴ e 1,7·10⁻³. Têm de 1 a 3 camadas e de 76 mil a 603 mil parâmetros: a arquitetura varia, o desempenho não.",
    "**Município importa:** o único trial sem embedding (d = 0) foi o pior de todos (70,23). d = 16 foi melhor que 4, 8 e 32 (mediana 68,92 contra 69,46–69,81). Com 32, o embedding tende a ajustar ruído dos municípios pequenos.",
    "**L2 forte causa subajuste:** os 4 trials com λ > 10⁻³ (todos entre 3,8·10⁻³ e 7,6·10⁻³) ficaram entre 69,24 e 70,81, com RMSE de treino tão alto quanto o de validação (69,0 a 71,1), a assinatura de **subajuste**. Isso acontece com dropout alto ou baixo e mostra que a ampliação do intervalo foi informativa: além de ~10⁻³ só piora. O TPE convergiu para L2 fraco (~10⁻⁵) e deixou dropout e LayerNorm fazerem a regularização.",
    "**Normalização:** LayerNorm venceu (mediana 68,91 em 26 trials) contra nenhuma (69,63) e BatchNorm (69,57), repetindo E2: a BatchNorm não ajuda com lotes grandes, e a LayerNorm estabiliza o treino e permite taxas de aprendizado maiores.",
    "**Dropout baixo é ruim:** todos os trials com dropout < 0,15 ficaram acima de 69,17.",
  ]));
  out.push(...figure(FIG("optuna_slices"), "RMSE de validação dos trials completos por hiperparâmetro (eixo y cortado no percentil 95)."));
  out.push(P(`**Overfitting nos trials.** Nos 10 melhores trials, a diferença entre RMSE de validação e de treino (medido em modo de avaliação, sem dropout) fica entre 0,58 e 1,01 ponto, e em todos os trials completos entre −0,26 e 2,65 (Figura ${nextFig()}). Com 400 mil exemplos, early stopping e dropout, nenhum trial completo chegou perto do overfitting visto em E1–E2. Os trials ruins são ruins por **subajuste** (L2 forte, sem município) ou otimização instável, não por memorização.`));
  out.push(...figure(FIG("optuna_gap"), "Diferença validação − treino nos trials completos, por dropout e normalização.", 0.75));

  out.push(H2("4.4 Seleção das arquiteturas para a avaliação final"));
  out.push(P(`O critério de seleção foi fixado **antes** de olhar qualquer resultado de teste e usa só o RMSE de validação do Optuna: (A) o melhor trial; (B) o de **menos parâmetros** entre os trials a até 0,5% do melhor RMSE; (C) o melhor trial com profundidade \"do outro lado\": como A tem 2 camadas, o melhor com 4 ou mais. A ideia é comparar, no mesmo protocolo, o melhor modelo, o mais barato equivalente e uma rede profunda. A Tabela ${nextTab()} lista as escolhas, e a Figura ${nextFig()} as localiza no plano custo × desempenho.`));
  const rows = Object.entries(sel).map(([k, m]) => {
    const p = m.params; const t = comp.find((r) => +r.number === m.trial);
    return [k.replace("_", " – "), `#${m.trial}`, `${m.hidden.join(" → ")}`, `${p.ativacao.toUpperCase()}; ${p.normalizacao === "layer" ? "LayerNorm" : p.normalizacao}; dropout ${n(p.dropout)}`,
      `L2 ${sci(p.l2)}; LR ${sci(p.lr)}; lote ${p.batch_size}; emb. ${p.emb_municipio}`, int(t.user_attrs_n_params), n(m.val_rmse_optuna)];
  });
  out.push(...table("Arquiteturas selecionadas (RMSE de validação no Optuna, treino de 400 mil).",
    ["Modelo", "Trial", "Camadas ocultas", "Ativação; norm.; dropout", "Otimização", "Parâmetros", "RMSE val."],
    rows, [0.13, 0.07, 0.17, 0.22, 0.21, 0.10, 0.10], ["l", "l", "l", "l", "l", "r", "r"], { size: 15 }));
  out.push(...figure(FIG("optuna_pareto"), "Trials completos no plano parâmetros × RMSE e duração × RMSE, com as três arquiteturas selecionadas."));
  out.push(P("A compacta (B) tem **9 vezes menos parâmetros** que A e treinou um trial em 9 s, contra 63 s, pagando 0,11 ponto de RMSE. A profunda (C), um funil de 4 camadas, fica 0,17 ponto atrás de A. Pelo platô observado, essas diferenças podem não sobreviver ao retreino no conjunto completo, e é isso que a próxima seção testa."));
  return out;
};
