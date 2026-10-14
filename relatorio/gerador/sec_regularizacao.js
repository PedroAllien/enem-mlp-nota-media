// Seção 4.2 – regularização
const { P, H2, H3, bullets, figure, table, nextFig, nextTab } = require("./lib");
const FIG = (f) => `${require("path").join(__dirname, "..", "..", "outputs")}/figures/${f}.png`;
const n = (x, k = 2) => Number(x).toLocaleString("pt-BR", { minimumFractionDigits: k, maximumFractionDigits: k });

const ROT = {
  sem_regularizacao: "sem regularização", early_stopping: "early stopping (paciência 10)", "dropout_0.2": "dropout 0,2",
  "dropout_0.5": "dropout 0,5", "l2_1e-4": "L2 λ = 10⁻⁴", "l2_1e-3": "L2 λ = 10⁻³", "l1_1e-5": "L1 λ = 10⁻⁵", "l1_1e-4": "L1 λ = 10⁻⁴",
  batchnorm: "BatchNorm", layernorm: "LayerNorm", dropout_l2_es: "dropout 0,3 + L2 10⁻⁴ + early stopping",
};
const ORDEM = Object.keys(ROT);

module.exports = function regularizacao(T, D) {
  // T: outputs/tables/regularizacao.csv ; D: outputs/tables/diagnostico_l2.csv
  const g = (esc, v) => T.find((r) => r.escala === esc && r.variante === v);
  const out = [];
  out.push(H2("4.2 Estratégias de regularização (E2)"));
  out.push(P(`Partimos da MLP 4×512 (924 mil parâmetros), a de maior overfitting em E1, e mudamos uma coisa por vez. Todas treinam o mesmo número de épocas com LR constante; as variantes com early stopping param antes e devolvem os pesos da melhor época. A Tabela ${nextTab()} traz, para cada variante, o melhor RMSE de validação, o estado na última época e o RMSE do **modelo devolvido**, que é o que importa na prática.`));
  const rows = [];
  for (const esc of ["n20k", "n500k"]) for (const v of ORDEM) {
    const r = g(esc, v);
    rows.push([esc === "n20k" ? "20 mil" : "500 mil", ROT[v], r.melhor_epoca, n(r.val_rmse_melhor), n(r.treino_rmse_final), n(r.val_rmse_final), n(r.gap_final), n(r.val_restaurado), Math.round(r.tempo_treino_s)]);
  }
  out.push(...table("Regularização na MLP 4×512 (validação: 100 mil exemplos de 2022; RMSE em pontos; tempo total em segundos).",
    ["Treino", "Variante", "Melhor época", "Val. (melhor)", "Treino (final)", "Val. (final)", "Gap final", "Modelo devolvido", "Tempo (s)"],
    rows, [0.08, 0.27, 0.08, 0.09, 0.09, 0.09, 0.09, 0.11, 0.10], ["l", "l", "r", "r", "r", "r", "r", "r", "r"], { size: 15 }));

  const s5 = g("n500k", "sem_regularizacao"), l2 = g("n500k", "l2_1e-3"), l1 = g("n500k", "l1_1e-4"), d5 = g("n500k", "dropout_0.5");
  const d2 = g("n500k", "dropout_0.2"), es5 = g("n500k", "early_stopping"), bn5 = g("n500k", "batchnorm"), ln5 = g("n500k", "layernorm");
  const s20 = g("n20k", "sem_regularizacao"), es20 = g("n20k", "early_stopping"), cb20 = g("n20k", "dropout_l2_es"), bn20 = g("n20k", "batchnorm");
  out.push(H3("Com 500 mil exemplos"));
  out.push(P(`Sem regularização, a rede passa do ponto ótimo na época ${s5.melhor_epoca} e termina com ${n(s5.treino_rmse_final, 1)} no treino e ${n(s5.val_rmse_final, 1)} na validação (Figura ${nextFig()}). As estratégias se dividem em três grupos:`));
  out.push(...bullets([
    `**Eliminam o overfitting:** L2 com λ = 10⁻³ e L1 com λ = 10⁻⁴ deixam treino e validação praticamente juntos até a última época (gaps de ${n(l2.gap_final, 1)} e ${n(l1.gap_final, 1)}) e dão os **melhores RMSE do experimento** (${n(l2.val_rmse_melhor)} e ${n(l1.val_rmse_melhor)}, nas épocas ${l2.melhor_epoca} e ${l1.melhor_epoca}). Dropout 0,5 chega perto (gap ${n(d5.gap_final, 1)}; ${n(d5.val_rmse_final)} na última época). Essas variantes trocam a memorização por um ajuste lento e estável, e o resultado deixa de depender de quando o treino para. Os valores mais fracos (L2 10⁻⁴, L1 10⁻⁵, dropout 0,2) só atrasam o overfitting (gaps de 5 a 13 pontos).`,
    `**Não regularizam, mas salvam o modelo:** o early stopping não impede o overfitting, só o interrompe. Ele devolve ${n(es5.val_restaurado)} (época ${es5.melhor_epoca}) e é a opção **mais barata** (${Math.round(es5.tempo_treino_s)} s, contra ${Math.round(s5.tempo_treino_s)} s sem regularização e ${Math.round(l2.tempo_treino_s)}–${Math.round(l1.tempo_treino_s)} s com L2/L1, cuja penalidade é recalculada a cada lote).`,
    `**Não ajudam:** a BatchNorm sofre overfitting tanto quanto a rede sem nada (${n(bn5.val_rmse_final, 1)} na última época), tem o pior melhor-ponto (${n(bn5.val_rmse_melhor)}) e é a mais lenta (${Math.round(bn5.tempo_treino_s)} s). Com lotes de 2048, o ruído das estatísticas do lote, que dá à BatchNorm algum efeito regularizador, é desprezível. A LayerNorm suaviza a curva, mas termina com gap de ${n(ln5.gap_final, 1)}. Normalização aqui é ferramenta de **otimização**, não de regularização.`,
  ]));
  out.push(P(`No **melhor ponto**, porém, todas as variantes de 500 mil ficam entre ${n(Math.min(...T.filter(r => r.escala === "n500k").map(r => +r.val_rmse_melhor)))} e ${n(Math.max(...T.filter(r => r.escala === "n500k").map(r => +r.val_rmse_melhor)))} pontos. A regularização praticamente não melhora o erro atingível: ela torna o treino **robusto**, sem exigir o momento exato de parar, e permite treinar por mais épocas com ganho pequeno (~0,3 ponto com L1/L2).`));
  out.push(...figure(FIG("reg_curvas_n500k"), "Curvas de treino e validação por estratégia, 500 mil exemplos (MLP 4×512)."));

  out.push(H3("Com 20 mil exemplos"));
  out.push(P(`Com poucos dados o quadro muda (Figura ${nextFig()}). Sem regularização, a rede decora o treino (RMSE ${n(s20.treino_rmse_final, 1)}) e termina com ${n(s20.val_rmse_final, 1)} na validação. Só as variantes com **early stopping** devolvem um modelo bom (${n(es20.val_restaurado)} e ${n(cb20.val_restaurado)}): o ótimo ocorre na 2ª–4ª época, antes de qualquer regularizador fazer diferença. Dropout reduz a memorização (validação final de 76–78), mas não a evita em 120 épocas. A BatchNorm, com lotes de 256 e pouca variedade de exemplos, piora até o melhor ponto (${n(bn20.val_rmse_melhor)}).`));
  out.push(P("Contrariando a intuição, **L1 e L2 deixaram a validação final pior** que sem regularização (83,6 a 98,3 contra 79,2), mesmo com o RMSE de treino subindo. Investigamos duas hipóteses (Tabela " + nextTab() + "; `scripts/03b_diagnostico_l2.py`):"));
  out.push(...bullets([
    "**(1) O embedding do município, que não é penalizado, compensaria os pesos encolhidos.** Descartada: sem embedding (d = 0), L2 10⁻³ e L1 10⁻⁴ continuam piores que nada (90,8 e 85,6 contra 82,9).",
    "**(2) Penalidade somada à perda perde efeito no Adam** (o passo adaptativo normaliza o gradiente da penalidade; Loshchilov e Hutter, 2019). Descartada como explicação: o *weight decay* desacoplado (AdamW, 0,01 a 1,0) também não segura a memorização (79,6 a 85,5).",
  ]));
  const drows = D.map((r) => [r.variante.replace("sem_regularizacao", "sem regularização").replace("adamw_wd", "AdamW wd = ").replace("l2_1e-3", "L2 λ = 10⁻³").replace("l1_1e-4", "L1 λ = 10⁻⁴"),
    r.embedding, r.otimizador, n(r.val_melhor), n(r.treino_final), n(r.val_final)]);
  out.push(...table("Diagnóstico das penalidades com 20 mil exemplos (MLP 4×512, 120 épocas, LR constante).",
    ["Variante", "Embedding (d)", "Otimizador", "Val. (melhor)", "Treino (final)", "Val. (final)"],
    drows, [0.30, 0.13, 0.13, 0.15, 0.14, 0.15], ["l", "r", "l", "r", "r", "r"], { size: 15 }));
  out.push(P("Nossa leitura: com 924 mil parâmetros para 20 mil exemplos, existem soluções de **norma pequena que ainda interpolam o ruído** do treino. Penalizar a norma muda qual dessas soluções é encontrada, e às vezes leva a uma que generaliza pior, mas não impede a interpolação. Com 500 mil exemplos, o ruído não pode mais ser interpolado com norma pequena, e a mesma penalidade passa a funcionar como esperado. Lição prática: **a força certa da penalidade depende do tamanho do conjunto de treino**, e com poucos dados o early stopping é a defesa principal. Por isso o early stopping está presente em todos os trials do Optuna e nos modelos finais."));
  out.push(...figure(FIG("reg_curvas_n20k"), "Curvas de treino e validação por estratégia, 20 mil exemplos (MLP 4×512)."));
  out.push(...figure(FIG("reg_resumo"), "RMSE de validação do modelo devolvido por estratégia e escala.", 0.85));
  return out;
};
