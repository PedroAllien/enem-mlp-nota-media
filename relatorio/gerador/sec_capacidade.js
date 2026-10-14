// Seção 4.1 – capacidade
const { P, H2, H3, bullets, figure, table, nextFig, nextTab } = require("./lib");
const FIG = (f) => `${require("path").join(__dirname, "..", "..", "outputs")}/figures/${f}.png`;
const n = (x, k = 2) => Number(x).toLocaleString("pt-BR", { minimumFractionDigits: k, maximumFractionDigits: k });
const int = (x) => Number(x).toLocaleString("pt-BR");

module.exports = function capacidade(T) {
  // T: linhas de outputs/tables/capacidade.csv
  const get = (esc, p, w) => T.find((r) => r.escala === esc && +r.profundidade === p && +r.largura === w);
  const out = [];
  out.push(H2("4.1 Profundidade e largura (E1)"));
  out.push(P(`A Tabela ${nextTab()} resume as 32 redes da grade. Lembrando o protocolo: nenhuma regularização, LR constante e todas as épocas treinadas. A coluna \"melhor\" é o RMSE de validação na melhor época (o que o early stopping devolveria); \"final\" é o RMSE na última época.`));
  const rows = [];
  for (const esc of ["n20k", "n500k"]) {
    for (const [p, w] of [[0, 0], [1, 32], [1, 128], [1, 512], [2, 32], [2, 128], [2, 512], [3, 32], [3, 128], [3, 512], [4, 32], [4, 128], [4, 512], [6, 32], [6, 128], [6, 512]]) {
      const r = get(esc, p, w);
      rows.push([esc === "n20k" ? "20 mil" : "500 mil", p === 0 ? "linear" : `${p} × ${w}`, int(r.n_params), r.melhor_epoca,
        n(r.val_rmse_melhor), n(r.treino_rmse_final), n(r.val_rmse_final), n(r.gap_final), n(r.tempo_epoca_s)]);
    }
  }
  out.push(...table("Grade de capacidade (validação: 100 mil exemplos de 2022). RMSE em pontos.",
    ["Treino", "Arquitetura", "Parâmetros", "Melhor época", "Val. (melhor)", "Treino (final)", "Val. (final)", "Gap final", "s/época"],
    rows, [0.09, 0.12, 0.12, 0.09, 0.11, 0.11, 0.11, 0.12, 0.13], ["l", "l", "r", "r", "r", "r", "r", "r", "r"], { size: 15 }));

  const lin20 = get("n20k", 0, 0), lin500 = get("n500k", 0, 0);
  const best500 = T.filter((r) => r.escala === "n500k" && +r.profundidade > 0).sort((a, b) => a.val_rmse_melhor - b.val_rmse_melhor)[0];
  const best20 = T.filter((r) => r.escala === "n20k" && +r.profundidade > 0).sort((a, b) => a.val_rmse_melhor - b.val_rmse_melhor)[0];
  const w512 = get("n500k", 4, 512), w32 = get("n500k", 4, 32);
  out.push(P(`**1) No melhor ponto, a arquitetura importa pouco.** Com 500 mil exemplos, todas as MLPs ficam entre ${n(Math.min(...T.filter(r => r.escala === "n500k" && +r.profundidade > 0).map(r => +r.val_rmse_melhor)))} e ${n(Math.max(...T.filter(r => r.escala === "n500k" && +r.profundidade > 0).map(r => +r.val_rmse_melhor)))} pontos de RMSE de validação (Figura ${nextFig()}), contra ${n(lin500.val_rmse_melhor)} da regressão linear. Ou seja, **uma única camada oculta já captura quase todo o ganho da não linearidade** (~${n(lin500.val_rmse_melhor - best500.val_rmse_melhor, 1)} ponto). Esse ganho vem de interações entre as entradas (por exemplo, o efeito da renda depende da UF e do tipo de escola), que um modelo aditivo não representa. Aprofundar ou alargar a rede não ajuda: a largura 512 fica sistematicamente um pouco pior que 32 e 128.`));
  out.push(...figure(FIG("cap_melhor_val"), "RMSE de validação na melhor época por número de camadas ocultas e largura. 0 camadas = regressão linear (mesmo ponto para as três larguras)."));
  out.push(P(`**2) Quantidade de dados importa mais que capacidade.** Passar de 20 mil para 500 mil exemplos reduz o melhor RMSE de ${n(best20.val_rmse_melhor)} (${best20.profundidade}×${best20.largura}) para ${n(best500.val_rmse_melhor)} (${best500.profundidade}×${best500.largura}), mais do que qualquer mudança de arquitetura dentro de uma mesma escala. Isso condiz com a Seção 2.3: o sinal disponível é limitado e ruidoso (desvio de ~73 pontos entre pessoas com o mesmo perfil), e mais exemplos ajudam a estimar melhor as médias condicionais, sobretudo as dos 1,7 mil municípios no embedding.`));
  out.push(P(`**3) Sem regularização, a largura determina o overfitting** (Figuras ${nextFig()} e ${nextFig() + 1}). Com 20 mil exemplos, as MLPs com 128 ou mais neurônios por camada chegam a RMSE de treino entre 5 e 21 pontos, ou seja, **decoram o conjunto de treino**, enquanto a validação sobe de ~71 para 79–100 pontos. Até a regressão linear sofre overfitting leve (gap de ${n(lin20.gap_final, 1)}), por causa dos ~28 mil parâmetros do embedding de município estimados com 20 mil exemplos. Com 500 mil exemplos, as redes de largura 32 quase não sofrem overfitting (gap ≈ 1 ponto em 30 épocas, independentemente da profundidade), enquanto as de largura 512 voltam a decorar: a ${w512.profundidade}×512 termina com ${n(w512.treino_rmse_final, 1)} no treino e ${n(w512.val_rmse_final, 1)} na validação, contra ${n(w32.treino_rmse_final, 1)}/${n(w32.val_rmse_final, 1)} da ${w32.profundidade}×32. O número de parâmetros cresce com o quadrado da largura, mas só linearmente com a profundidade, e é isso que domina.`));
  out.push(...figure(FIG("cap_gap"), "Diferença entre RMSE de validação e de treino na última época. Valores altos = overfitting."));
  out.push(...figure(FIG("cap_curvas"), "Curvas de treino e validação para quatro capacidades nas duas escalas. O ponto marca a melhor época da validação."));
  out.push(P(`**4) Quanto maior a capacidade, mais cedo aparece o melhor ponto.** Com 500 mil exemplos, as redes de largura 512 atingem o mínimo da validação nas épocas 4–6, e as de largura 32 nas épocas 15–21. Com 20 mil, nas épocas 2–3 e 6–11. Nas maiores, as curvas de treino com LR constante mostram picos de instabilidade (6×512 com 20 mil exemplos), um sinal de taxa de aprendizado alta demais para aquela capacidade sem normalização. Nenhuma configuração ficou em **subajuste** claro: mesmo a 1×32 tem treino e validação praticamente iguais e no patamar das maiores. O único modelo limitado pela capacidade é o linear, cerca de 1 ponto pior que qualquer MLP com 500 mil exemplos e com treino ≈ validação (${n(lin500.treino_rmse_final, 1)} × ${n(lin500.val_rmse_final, 1)}), a assinatura de viés de modelo.`));
  out.push(P(`**5) Custo.** O tempo por época cresce com o número de parâmetros (Figura ${nextFig()}): com 500 mil exemplos, de ${n(lin500.tempo_epoca_s)} s (linear) a ${n(get("n500k", 6, 512).tempo_epoca_s)} s (6×512), sem nenhum ganho de RMSE nas maiores. A 6×512 tem ${int(get("n500k", 6, 512).n_params)} parâmetros, ${Math.round(get("n500k", 6, 512).n_params / get("n500k", 1, 32).n_params)} vezes a 1×32, e é ${n(get("n500k", 6, 512).val_rmse_melhor - get("n500k", 1, 32).val_rmse_melhor)} ponto pior. Parte fixa importante: o embedding de município (16 × 1.742 ≈ 28 mil parâmetros) é a maior parte dos parâmetros das redes pequenas.`));
  out.push(...figure(FIG("cap_custo"), "Segundos por época (500 mil exemplos, GPU M2) por número de parâmetros.", 0.7));
  return out;
};
