// Seções 1-3: introdução, dados e metodologia (não dependem dos resultados).
const { d, P, H1, H2, H3, bullets, figure, table, nextFig, nextTab } = require("./lib");
const FIG = (f) => `${require("path").join(__dirname, "..", "..", "outputs")}/figures/${f}.png`;

module.exports = function parte1(R) {
  const out = [];

  // ---------- 1 ----------
  out.push(H1("1. Introdução e objetivo"));
  out.push(P("O Exame Nacional do Ensino Médio (ENEM) é a principal porta de entrada para o ensino superior público no Brasil (SiSU) e também é usado pelo ProUni e pelo Fies. Em 2022 foram 3,48 milhões de inscritos e, em 2023, 3,93 milhões. Já se sabe que o desempenho no exame acompanha de perto as condições socioeconômicas e escolares do participante. A pergunta deste trabalho é **quanto do desempenho dá para antecipar só com o que se sabe na inscrição**, meses antes da prova."));
  out.push(P("**Objetivo.** Prever a **nota média** de um participante no ENEM usando apenas informações disponíveis antes da prova: tipo de escola, situação de conclusão do ensino médio, renda familiar, escolaridade dos responsáveis, acesso a computador e internet, município e estado, características da residência e faixa etária."));
  out.push(P("**Utilidade da previsão.** (i) Identificar, antes da prova, grupos e regiões com desempenho esperado baixo, para direcionar ações de apoio (cursinhos, reforço, material); (ii) medir quanto do resultado já é explicado pelo perfil de entrada; (iii) servir de referência para avaliar programas e escolas pelo desempenho **acima do esperado** dado o perfil dos alunos. O modelo não foi feito para tomar decisões sobre um indivíduo, e a Seção 6 discute por que ele não deve ser usado assim."));
  out.push(P("**Definição do problema.** A unidade de observação é o **participante de uma edição do ENEM**. A variável alvo, `NOTA_MEDIA`, é a média aritmética simples das cinco notas: Ciências da Natureza (CN), Ciências Humanas (CH), Linguagens e Códigos (LC), Matemática (MT) e Redação, em pontos na escala do ENEM. Como o alvo é contínuo, o problema é de **regressão**. A frente escolhida é a **3.1 – Multilayer Perceptrons**: os dados são tabulares, sem estrutura espacial ou temporal que justifique convolução ou recorrência, e a MLP é a arquitetura profunda natural para esse caso."));

  // ---------- 2 ----------
  out.push(H1("2. Fonte e caracterização dos dados"));
  out.push(H2("2.1 Fonte e escolha das edições"));
  out.push(P("Os dados são os **Microdados do ENEM** publicados pelo INEP e indexados no Portal Brasileiro de Dados Abertos (https://dados.gov.br/dados/conjuntos-dados/inep-microdados-do-enem). Cada edição é um CSV de ~1,8 GB com 76 colunas e uma linha por inscrito: dados cadastrais, local de prova, presença, notas, respostas e o questionário socioeconômico preenchido na inscrição."));
  out.push(P("Usamos as edições **2022 e 2023**. A de 2023 é a mais recente em que questionário e notas estão no mesmo registro. A partir de 2024, o INEP passou a publicar `PARTICIPANTES` (questionário) e `RESULTADOS` (notas) em arquivos separados que, segundo o próprio Leia-Me, *não possuem chave de ligação em comum*. Isso impede o cruzamento necessário para este problema. Com duas edições, dá para treinar num ano e testar no ano seguinte (Seção 3.1), que é o uso realista do modelo."));
  out.push(P("**Filtragem.** Mantivemos quem esteve presente nas quatro provas objetivas e tem as cinco notas. Faltantes e eliminados não têm nota média definida, e prever presença seria outro problema. Também removemos registros com nota 0 numa prova objetiva apesar da presença (2.910 em 2022 e 19.501 em 2023), porque a TRI não gera 0 para quem respondeu à prova e esses casos indicam registro anômalo. A **redação 0 foi mantida**: é um resultado legítimo pelas regras (em branco, fuga ao tema, cópia do texto motivador) e atinge 4,2% dos participantes de 2022."));
  out.push(...table("Funil de registros por edição.",
    ["Etapa", "2022", "% inscritos", "2023", "% inscritos"],
    [["Inscritos", "3.476.105", "100,0", "3.933.955", "100,0"],
     ["Presentes nas 4 provas objetivas", "2.344.823", "67,5", "2.678.264", "68,1"],
     ["Com as 5 notas", "2.344.823", "67,5", "2.678.264", "68,1"],
     ["Sem nota objetiva igual a 0 (base final)", "2.341.913", "67,4", "2.658.763", "67,6"]],
    [0.40, 0.15, 0.15, 0.15, 0.15], ["l", "r", "r", "r", "r"]));

  out.push(H2("2.2 Variáveis de entrada"));
  out.push(P(`As entradas seguem os grupos definidos no problema. Todas vêm da ficha de inscrição, preenchida meses antes da prova, ou do local de prova, definido antes do exame. A Tabela ${nextTab()} resume tipos, codificação e limitações.`));
  out.push(...table("Variáveis de entrada (todas conhecidas antes da prova).",
    ["Grupo", "Variável (microdados)", "Tipo / unidade", "Limitações"],
    [["Tipo de escola", "`TP_ESCOLA`; `TP_DEPENDENCIA_ADM_ESC`; `TP_LOCALIZACAO_ESC`", "nominal (3, 4 e 2 categorias)", "`TP_ESCOLA` só é respondido por concluintes do ano (59% \"não respondeu\"); campos da escola ausentes para 70%"],
     ["Situação de conclusão do EM", "`TP_ST_CONCLUSAO`; `TP_ANO_CONCLUIU`", "nominal (4); ordinal, anos desde a conclusão", "código 16 harmonizado entre anos (ver texto)"],
     ["Renda familiar", "`Q006`", "ordinal, 17 faixas em salários mínimos", "autodeclarada; faixas largas no topo"],
     ["Escolaridade dos responsáveis", "`Q001` (pai), `Q002` (mãe)", "ordinal de 7 níveis + \"não sei\"", "\"não sei\": 8,6% (pai), 2,7% (mãe)"],
     ["Computador e internet", "`Q024`, `Q025`", "ordinal (0 a 4+); binária", "autodeclarada"],
     ["Município e estado", "`CO_MUNICIPIO_PROVA`; `SG_UF_PROVA`", "nominal (1.747 municípios; 27 UFs)", "município da **prova**: o de residência foi retirado dos microdados (LGPD)"],
     ["Características da residência", "`Q005` (moradores); `Q007`–`Q023` (cômodos e bens)", "numérica (1–20); ordinais", "autodeclaradas; alguns bens perderam sentido (DVD, telefone fixo)"],
     ["Faixa etária", "`TP_FAIXA_ETARIA`", "ordinal, 20 faixas", "idade exata retirada pelo INEP (LGPD)"]],
    [0.17, 0.27, 0.24, 0.32], ["l", "l", "l", "l"], { size: 16 }));
  out.push(P("**Harmonização entre anos.** `TP_ANO_CONCLUIU` conta os anos entre a conclusão do ensino médio e a prova (1 = ano anterior), mas o último código é \"antes de 2007\", que vale 16 em 2022 e 17 em 2023. Agrupamos ambos em 16 = \"16 anos ou mais\", para que cada código signifique a mesma coisa nos dois anos. Sem isso, 82.896 participantes de 2023 teriam uma categoria nunca vista no treino. As faixas de renda (`Q006`) são definidas em múltiplos do salário mínimo de cada ano e, por isso, são comparáveis entre edições."));

  out.push(H3("Variáveis excluídas e risco de vazamento"));
  out.push(...table("Variáveis deixadas de fora e motivo.",
    ["Variáveis", "Motivo"],
    [["`NU_NOTA_*` (as 4 provas e a redação), `NU_NOTA_COMP1–5`", "compõem o alvo (vazamento direto)"],
     ["`TX_RESPOSTAS_*`, `TX_GABARITO_*`", "respostas e gabarito da própria prova"],
     ["`TP_PRESENCA_*`, `TP_STATUS_REDACAO`", "só existem depois da prova; usadas apenas no filtro"],
     ["`CO_PROVA_*`", "cor do caderno, atribuída no dia da prova"],
     ["`TP_SEXO`, `TP_COR_RACA`", "conhecidas antes da prova, mas excluídas de propósito para não usar atributos protegidos como preditores; usadas só na auditoria de erro por grupo (Seção 5.3)"],
     ["`Q003`, `Q004` (ocupação dos pais), `TP_LINGUA`, `TP_ESTADO_CIVIL`, `TP_NACIONALIDADE`", "conhecidas antes da prova, mas fora do escopo definido; a ocupação dos pais é muito redundante com escolaridade e renda"]],
    [0.42, 0.58], ["l", "l"], { size: 16 }));

  out.push(H2("2.3 Estatísticas descritivas, ausentes, duplicados e anomalias"));
  out.push(P("**Ausentes.** Entre os participantes válidos, o questionário socioeconômico **não tem valores ausentes** (0% em todas as perguntas, nos dois anos). A ausência que existe é **estrutural**. Os campos da escola (`TP_DEPENDENCIA_ADM_ESC`, `TP_LOCALIZACAO_ESC`) só são preenchidos para quem conclui o ensino médio no ano da prova (70,4% ausentes em 2022). `TP_ESCOLA` é \"não respondeu\" para todos que já concluíram ou não estão cursando o EM (59,2%). `TP_ANO_CONCLUIU` é \"não informado\" para 62,3%. Como a ausência depende da situação do participante, e não do acaso, ela foi mantida como **categoria própria** em vez de imputada: o modelo pode usar o fato de o campo estar vazio."));
  out.push(P("**Duplicados.** Não há linhas duplicadas considerando entradas e as cinco notas (0 casos). Apenas 0,7% dos participantes repetem exatamente o perfil de entrada de outro, porque o questionário tem muitas combinações possíveis. Nos 498 perfis repetidos com 5 ou mais pessoas, o desvio-padrão da nota entre pessoas **idênticas para o modelo** é de 72,6 pontos. Isso é um sinal direto de erro irredutível alto: nenhum modelo baseado nessas entradas acerta notas diferentes para entradas iguais."));
  out.push(P(`**Anomalias.** Além dos registros com nota objetiva 0, já removidos, só ~0,5% das notas médias fica fora de 1,5·IQR. São notas reais (redação 0 puxando a média para baixo; notas máximas) e foram mantidas. A Figura ${nextFig()} mostra como a nota média varia com as principais entradas.`));
  out.push(...figure(FIG("eda_media_por_categoria"), "Nota média por categoria das principais entradas (2022). A linha cinza é a média geral (543,7). Renda, computadores e escolaridade da mãe têm relação quase monotônica com a nota; \"não sei\" (H) na escolaridade da mãe fica abaixo de todos os níveis, menos \"nunca estudou\"."));
  out.push(P(`Para comparar a força de cada entrada isolada, usamos a **razão de correlação η²**, a fração da variância do alvo explicada pelas médias por categoria (Figura ${nextFig()}). A renda familiar (0,22), o número de computadores (0,19) e a escolaridade do pai e da mãe (0,14) lideram; o município da prova (0,10) vem logo depois e explica o dobro da UF (0,05), porque a nota varia em média 75 pontos entre municípios de uma mesma UF. A situação de conclusão sozinha explica quase nada (0,004), mas interage com outras variáveis: é ela que define se os campos da escola existem.`));
  out.push(...figure(FIG("eda_eta2"), "Razão de correlação η² entre cada entrada e a nota média (2022). Medida univariada: não considera redundância entre variáveis.", 0.78));

  out.push(H2("2.4 Distribuição da variável alvo"));
  out.push(...table("Estatísticas da nota média (pontos).",
    ["Estatística", "2022", "2023"],
    [["Participantes", "2.341.913", "2.658.763"], ["Média", "543,7", "542,4"], ["Desvio-padrão", "87,7", "93,0"],
     ["Mínimo / máximo", "254,6 / 856,0", "246,4 / 862,6"], ["Percentis 1 / 25 / 50 / 75 / 99", "332 / 485 / 541 / 602 / 750", "326 / 477 / 540 / 607 / 753"],
     ["Assimetria", "0,05", "0,04"], ["Curtose (excesso)", "−0,06", "−0,27"], ["% fora de 1,5·IQR", "0,47", "0,13"]],
    [0.46, 0.27, 0.27], ["l", "r", "r"]));
  out.push(P(`O alvo é **unimodal e quase simétrico** (Figura ${nextFig()}), com faixa de ~250 a ~860 pontos. Por isso não foi preciso transformar o alvo (log, Box-Cox); bastou padronizá-lo para o treino. A dispersão de 2023 é maior (93,0 contra 87,7 pontos), o que, por si só, tende a aumentar o RMSE no teste de 2023.`));
  out.push(...figure(FIG("eda_alvo_hist"), "Distribuição da nota média em 2022 e 2023 (linhas verticais = médias).", 0.85));
  out.push(P(`**Por que a média das cinco notas?** As notas das cinco provas têm correlação positiva entre si, de 0,43 (CN × Redação) a 0,74 (CH × LC). Isso indica um fator comum de desempenho, que a média resume e que é o principal critério de seleção no SiSU. A redação é a nota mais \"ruidosa\" (discreta, em múltiplos de 20, com massa em 0; Figura ${nextFig()}) e pesa 20% no alvo.`));
  out.push(...figure(FIG("eda_notas_por_prova"), "Distribuição de cada nota em 2022."));

  out.push(H2("2.5 Mudança de distribuição entre 2022 e 2023"));
  out.push(P(`As médias quase não mudam entre os anos (nota média −1,3 ponto; Matemática −5,4; CN +2,8), e as entradas também são estáveis: a maior distância de variação total entre as distribuições de uma entrada em 2022 e 2023 é 0,06 (computadores), seguida da renda (0,05; Figura ${nextFig()}). Ou seja, o teste de 2023 mede principalmente se o **mapeamento perfil → nota** se mantém, e não uma mudança forte na população.`));
  out.push(...figure(FIG("eda_renda_2022_2023"), "Distribuição da renda familiar declarada (Q006) em 2022 e 2023.", 0.85));

  // ---------- 3 ----------
  out.push(H1("3. Metodologia"));
  out.push(H2("3.1 Separação entre treino, validação e teste"));
  out.push(...table("Partições usadas.",
    ["Partição", "Origem", "Participantes", "Uso"],
    [["Treino", "ENEM 2022 (80%)", "1.873.529", "ajuste dos pesos e de todo o pré-processamento"],
     ["Validação", "ENEM 2022 (10%)", "234.192", "early stopping, LR scheduler, Optuna e seleção de modelos"],
     ["Teste-2022", "ENEM 2022 (10%)", "234.192", "desempenho no mesmo ano (avaliado uma vez, no fim)"],
     ["Teste-2023", "ENEM 2023 (100%)", "2.658.763", "desempenho no ano seguinte (avaliado uma vez, no fim)"]],
    [0.15, 0.2, 0.17, 0.48], ["l", "l", "r", "l"]));
  out.push(P("A divisão de 2022 é aleatória e **estratificada por decis do alvo** (semente 42), para que as três partes tenham a mesma distribuição de notas. Cada participante aparece uma única vez nos microdados, então não há o mesmo indivíduo em duas partições. Consideramos dividir por município, como o enunciado sugere para unidades com várias observações, mas aqui o município é uma **característica de entrada**, não uma unidade medida repetidamente. Dividir por município responderia a outra pergunta (prever para municípios nunca vistos) e impediria o modelo de aprender o efeito municipal, que é uma das entradas pedidas. A verificação de generalização mais forte é o **teste temporal**: o modelo nunca vê 2023 e é avaliado em todos os 2,66 milhões de participantes daquele ano. Municípios com menos de 30 participantes no treino, e os que aparecem só em 2023 (1.478 participantes), caem numa categoria \"raro/novo\"."));

  out.push(H2("3.2 Pré-processamento"));
  out.push(P("Todos os parâmetros de transformação foram **estimados só no treino** e depois aplicados à validação e aos testes:"));
  out.push(...bullets([
    "**Categóricas (29 variáveis) → one-hot**, 191 colunas no total. Ausência e \"não sei\" são categorias próprias. Uma categoria nunca vista no treino vira um bloco de zeros (não ocorreu depois da harmonização). Preferimos one-hot a codificação ordinal porque várias variáveis misturam níveis ordenados com respostas não ordenáveis (\"não sei\", \"não respondeu\") e porque a MLP pode aprender efeitos não monotônicos (Figura 1, escolaridade da mãe). O custo é pequeno: 191 colunas.",
    "**Numérica (`Q005`, nº de moradores) → z-score** com média e desvio do treino, mais um indicador de ausência (2 colunas).",
    "**Município → índice inteiro → camada de embedding.** Com 1.742 índices, o one-hot seria uma matriz quase toda de zeros. Um embedding de dimensão *d* é matematicamente uma camada linear sem viés aplicada ao one-hot do município, ou seja, continua sendo uma MLP. *d* é um hiperparâmetro (0 a 32; 0 = sem município).",
    "**Alvo → z-score** (μ = 543,72; σ = 87,74 no treino). As métricas são sempre calculadas depois de desfazer a padronização, em pontos.",
    "**Desbalanceamento:** não se aplica (regressão). A estratificação por decis garante a mesma distribuição de notas em treino, validação e teste-2022.",
  ]));
  out.push(P("Para economizar memória, o one-hot não é materializado (1,87 milhão × 191 floats ≈ 1,4 GB). Guardamos só a posição do \"1\" de cada variável, e o vetor one-hot é montado na GPU dentro do `forward`. O resultado é idêntico ao one-hot convencional."));

  out.push(H2("3.3 Arquitetura"));
  out.push(P(`A Figura ${nextFig()} mostra a MLP. A entrada concatena o one-hot (191), as numéricas (2) e o embedding do município (*d*). Seguem *L* blocos ocultos, cada um com Linear → normalização opcional (BatchNorm ou LayerNorm) → ativação (ReLU, GELU ou SiLU) → Dropout. A saída é **uma camada linear com um neurônio e sem ativação**, como é adequado para regressão sem limites rígidos (a nota pode ir de ~250 a ~860). Com *L* = 0 o modelo vira uma **regressão linear** sobre as mesmas entradas, que usamos como referência. A largura pode ser constante ou em \"funil\" (cada camada com metade da anterior).`));
  out.push(...figure(FIG("arquitetura"), "Arquitetura da MLP. O bloco oculto é repetido L vezes; normalização, ativação, dropout, L, largura e d são hiperparâmetros."));

  out.push(H2("3.4 Treinamento"));
  out.push(...bullets([
    "**Perda:** erro quadrático médio (MSE) no alvo padronizado, mais penalidades explícitas λ₁‖W‖₁ e λ₂‖W‖₂² sobre os pesos das camadas lineares (sem penalizar vieses, normalização e embedding). Penalidades explícitas, em vez do `weight_decay` do otimizador, deixam L1 e L2 comparáveis entre si.",
    "**Otimizador:** Adam, com lotes embaralhados a cada época.",
    "**Taxa de aprendizado:** `ReduceLROnPlateau` (multiplica por 0,5 quando a validação não melhora por 2 épocas) nos experimentos de busca e nos modelos finais. Nos estudos controlados de capacidade e regularização, a taxa é **constante**, para que a redução da taxa não mascare o overfitting.",
    "**Early stopping:** paciência de 5 épocas (4 no Optuna) sobre o RMSE de validação, com tolerância de 0,001 ponto, **restaurando os pesos da melhor época**.",
    "**Curvas de treino:** a cada época, o RMSE de treino é medido em modo de avaliação numa amostra fixa de 100 mil exemplos do treino. Assim dropout e BatchNorm não distorcem a comparação treino × validação.",
    "**Hardware:** Mac com chip Apple M2 (8 núcleos, 16 GB de memória unificada), PyTorch 2.14 com GPU via MPS. Os dados ficam inteiros na GPU.",
  ]));

  out.push(H2("3.5 Métricas"));
  out.push(P("A **métrica principal é o RMSE** (raiz do erro quadrático médio), em pontos: está na unidade do alvo, penaliza mais os erros grandes e é coerente com a perda MSE usada no treino. Complementamos com o **MAE** (erro absoluto médio, o erro \"típico\" em pontos, menos sensível a extremos), o **R²** (fração da variância explicada, comparável ao η² da Seção 2.3) e o **viés** (erro médio: previsto − real), importante no teste temporal. Referência: prever sempre a média do treino dá RMSE ≈ desvio-padrão do alvo, ou seja, ~88 pontos em 2022 e ~93 em 2023."));

  out.push(H2("3.6 Protocolo experimental"));
  out.push(...table("Experimentos realizados (todos com a mesma divisão de dados e o mesmo pré-processamento).",
    ["Exp.", "Pergunta", "Dados de treino", "Configuração"],
    [["E1 – Capacidade", "Como profundidade e largura afetam erro, overfitting e custo?", "20 mil e 500 mil exemplos (amostras fixas)", "L ∈ {0,1,2,3,4,6} × largura ∈ {32,128,512}; sem regularização; LR constante 10⁻³; 120 / 30 épocas"],
     ["E2 – Regularização", "Qual estratégia controla o overfitting e a que custo?", "20 mil e 500 mil", "MLP 4×512; 11 variantes, cada uma muda uma coisa: early stopping, dropout, L1, L2, BatchNorm, LayerNorm, combinação"],
     ["E3 – Optuna", "Qual combinação de hiperparâmetros minimiza o RMSE de validação?", "400 mil (validação: 100 mil)", "TPE multivariado + poda por mediana; 60 trials; 10 hiperparâmetros"],
     ["E4 – Modelos finais", "As 3 arquiteturas selecionadas generalizam para o teste e para 2023? A que custo?", "1,87 milhão (completo)", "3 sementes por modelo; referências: média e regressão linear"]],
    [0.16, 0.30, 0.18, 0.36], ["l", "l", "l", "l"], { size: 16 }));
  out.push(P("E1 e E2 usam duas escalas de treino de propósito. Com 20 mil exemplos, a capacidade da rede supera em muito a informação disponível e o overfitting fica evidente. Com 500 mil, aparece o regime em que estão os modelos finais. Os dois experimentos usam uma amostra fixa de 100 mil exemplos da validação."));
  return out;
};
