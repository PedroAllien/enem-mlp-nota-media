// Seção 4.3 (primeira parte) – espaço de busca e configuração do estudo
const { P, H2, H3, bullets, table, nextTab } = require("./lib");

module.exports = function optunaEspaco() {
  const out = [];
  out.push(H2("4.3 Busca de hiperparâmetros com Optuna (E3)"));
  out.push(H3("Espaço de busca e justificativa"));
  out.push(P(`O espaço (Tabela ${nextTab()}) foi definido a partir de práticas usuais para MLPs em dados tabulares e de um teste inicial de velocidade. E1 e E2 rodaram antes do estudo e serviram para checar se cada intervalo era informativo. Houve **um ajuste**: o intervalo da penalidade L2 era originalmente [10⁻⁷, 10⁻³], mas E2 (500 mil exemplos) mostrou o melhor λ exatamente no limite superior, então ampliamos para [10⁻⁶, 10⁻²] antes de iniciar o estudo. Esse ajuste usa só treino e validação; o teste não foi consultado.`));
  out.push(...table("Espaço de busca do Optuna.",
    ["Hiperparâmetro", "Intervalo / opções", "Escala", "Por que entra e por que esse intervalo"],
    [["Camadas ocultas", "1 a 6", "inteiro", "E1: 1 camada já captura o ganho não linear, e mais camadas não ajudaram sem regularização. A busca verifica se isso muda com dropout e normalização. A regressão linear (0) fica como referência separada."],
     ["Largura (1ª camada)", "32 a 1024", "log", "E1 cobriu 32–512; a busca vai até 1024 porque, com dropout ou normalização, redes largas podem deixar de decorar. Escala log porque o efeito da largura é multiplicativo (parâmetros ∝ largura²)."],
     ["Formato", "constante; funil (½ por camada)", "categórico", "o funil corta parâmetros das camadas profundas; é uma forma comum em dados tabulares de reduzir custo sem perder a largura na entrada"],
     ["Dropout", "0 a 0,5", "linear", "E2: reduziu o overfitting nas duas escalas (com 500 mil, gap final de 49 para 3 pontos com p = 0,5). Acima de 0,5 a rede passa a perder sinal útil."],
     ["Normalização", "nenhuma; BatchNorm; LayerNorm", "categórico", "E2: muda a estabilidade e a velocidade de convergência; o efeito no erro depende da combinação com LR e lote"],
     ["Penalidade L2 (λ₂)", "10⁻⁶ a 10⁻²", "log", "E2 (500 mil): λ = 10⁻³ eliminou o overfitting e deu o melhor RMSE do experimento, então o intervalo cobre uma década acima e três abaixo. L1 ficou de fora: em E2 teve efeito equivalente (68,96 × 68,97), e buscar as duas gastaria orçamento em hiperparâmetros redundantes."],
     ["Taxa de aprendizado", "10⁻⁴ a 3·10⁻³", "log", "faixa usual do Adam. E1 mostrou instabilidade com 10⁻³ constante nas redes grandes, e o limite superior atende redes pequenas com lotes grandes."],
     ["Tamanho do lote", "512; 1024; 2048; 4096", "categórico", "afeta o custo (vazão da GPU) e o ruído do gradiente (regularização implícita). Lotes menores que 512 deixam a época lenta demais com 400 mil exemplos."],
     ["Dimensão do embedding do município", "0; 4; 8; 16; 32", "categórico", "0 testa se o município acrescenta algo além da UF (η² 0,10 contra 0,05). Acima de 32 não faz sentido para ~1,7 mil categorias."],
     ["Ativação", "ReLU; GELU; SiLU", "categórico", "influência esperada pequena; entra com só 3 opções, barato para o TPE. Tanh e sigmoide ficaram de fora (saturação)."]],
    [0.17, 0.17, 0.10, 0.56], ["l", "l", "l", "l"], { size: 15 }));
  out.push(P("**Fixados, fora da busca:** otimizador Adam, perda MSE, `ReduceLROnPlateau` (fator 0,5, paciência 2), no máximo 25 épocas com early stopping de paciência 4. São escolhas-padrão com pouca influência esperada, ou cujo efeito já é coberto por outro hiperparâmetro (o scheduler compensa parte de uma LR inicial ruim)."));
  out.push(H3("Configuração do estudo"));
  out.push(...bullets([
    "**Métrica otimizada:** RMSE de validação (pontos) do modelo restaurado na melhor época, minimizado.",
    "**Dados:** amostra fixa de 400 mil exemplos do treino e 100 mil da validação. E1 mostrou que, a partir de ~500 mil, a arquitetura já está no regime de \"muitos dados\". Com 400 mil, cada trial leva ~1 min em vez de ~5 min com o treino completo. Os modelos selecionados são retreinados no treino completo em E4.",
    "**Amostrador:** TPE multivariado (modela dependências entre hiperparâmetros, como LR × lote e dropout × largura), com 12 trials iniciais aleatórios (20% do orçamento) para cobrir o espaço antes de explorar as regiões promissoras.",
    "**Poda:** `MedianPruner`. A partir da 4ª época, interrompe o trial cujo RMSE de validação está pior que a mediana dos trials anteriores na mesma época (só depois de 8 trials completos). Corta trials ruins cedo e economiza orçamento para os promissores.",
    "**Orçamento:** 60 trials. Com 10 hiperparâmetros, o TPE costuma estabilizar entre 40 e 80 trials, e o histórico (Figura adiante) mostra se a busca convergiu.",
    "**Ponto de partida:** o primeiro trial é uma configuração escolhida à mão (3×256, ReLU, dropout 0,1, LR 10⁻³, lote 2048, embedding 16), que serve de referência para o ganho da busca.",
    "**Registro:** o estudo fica num SQLite (`outputs/optuna.db`) com parâmetros, valores intermediários por época, estado (completo/podado), parâmetros treináveis, tempo e RMSE de treino de cada trial, e pode ser auditado ou retomado. A tabela completa está em `outputs/tables/optuna_trials.csv`.",
  ]));
  return out;
};
