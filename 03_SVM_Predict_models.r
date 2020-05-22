# Definindo o diretório de trabalho
getwd()
setwd("/Cap11_Classificacao_Regressao_Support_Vector_Machines_SVMS/R")


# Pacotes
install.packages("gains")
install.packages("pROC")
install.packages("ROSE")
install.packages("mice")
library(dplyr)
library(caret)
library(pROC)
library(e1071)
library(mice)
library(readr)

# Carregando os dados pré-processados
?read.csv
dados_treino1 <- read.csv("dados/dados_treino_balanceados.csv")
dados_teste1 <- read.csv("dados/dados_teste.csv")
dim(dados_treino1)
dim(dados_teste1)
View(dados_treino1)
View(dados_teste1)

# A função read_csv mostra o que aconteceu
?read_csv
dados_treino <- read_csv("dados/dados_treino_balanceados.csv")
dados_teste <- read_csv("dados/dados_teste.csv")
dim(dados_treino)
dim(dados_teste)
View(dados_treino)
View(dados_teste)

# Removemos a coluna X criada na indexação randômica
dados_treino <- dados_treino[-1]
dados_teste <- dados_teste[-1]

dim(dados_treino)
dim(dados_teste)

# Transformando a variável target em valor numérico
View(dados_treino)
str(dados_treino$Customer_cat)
View(dados_treino$Customer_cat)

# Transforma a variável em numérico
dados_treino$Customer_cat <- as.numeric(as.factor(dados_treino$Customer_cat))
str(dados_treino$Customer_cat)
View(dados_treino$Customer_cat)

# Transforma a variável em dado categórico
dados_treino$Customer_cat <- as.factor(dados_treino$Customer_cat)
str(dados_treino$Customer_cat)
View(dados_treino$Customer_cat)

# Transforma a variável do dataset de teste
dados_teste$Customer_cat <- as.numeric(as.factor(dados_teste$Customer_cat))
dados_teste$Customer_cat <- as.factor(dados_teste$Customer_cat)
str(dados_teste$Customer_cat)
View(dados_teste$Customer_cat)

################################################################################################################################################
                                                        ##### Modelagem Preditiva ##### 
################################################################################################################################################

################################################################################################################################################
                                  # Primeira versão do modelo SVM - Versão Padrão com Kernel Radial (RBF)
                                  # O algoritmo escolhe o tipo de SVM de acordo com o tipo de dado da variável target
################################################################################################################################################

?svm
# na.action: Escolhe a ação a ser tomada quando identificado valores NA.
# scale: padronização ou não dos dados
modelo_v1 <- svm(Customer_cat ~ ., data = dados_treino, na.action = na.omit, scale = TRUE)
summary(modelo_v1)
print(modelo_v1)

# Fazendo previsões com o modelo
previsoes_v1 <- predict(modelo_v1, newdata = dados_teste)

# Matriz de Confusão
?caret::confusionMatrix
caret::confusionMatrix(previsoes_v1, dados_teste$Customer_cat)

##############################################################################
# Troubleshooting: Dados de Teste e Treino com tamanho diferente.
##############################################################################

# Comprimento do valor real e do valor previsto
length(dados_teste$Customer_cat)
length(previsoes_v1)

# Verificando se existe valores NA no Dataset de teste
sum(is.na(dados_teste))

# Removemos valores NA
dados_teste = na.omit(dados_teste)
length(dados_teste$Customer_cat)
sum(is.na(dados_teste))

# E agora sim a matriz de confusão
caret::confusionMatrix(previsoes_v1, dados_teste$Customer_cat)

# Métricas utilizada para explicar o modelo como um todo! Sempre usar + de uma métrica para análise de resultado
install.packages("multiROC")
library(multiROC)

# Realiza o cálculo da curva ROC
?multiclass.roc
curva_roc <- multiclass.roc(response = dados_teste$Customer_cat, predictor = previsoes_v1)
class(dados_teste$Customer_cat)
class(previsoes_v1)

# Faz a conversão do dado da previsão em numérico
curva_roc <- multiclass.roc(response = dados_teste$Customer_cat, predictor = as.numeric(as.factor(previsoes_v1)))

# Score AUC (Area Under The Curve)
curva_roc$auc

# Juntando valores reais e previstos na mesma tabela

# Previsões
valores_previstos <- data.frame(as.numeric(as.factor(previsoes_v1)))
colnames(valores_previstos) <- ("previsão")

# Valores reais
valores_reais <- data.frame(as.numeric(as.factor(dados_teste$Customer_cat)))
colnames(valores_reais) <- ("valor_real")

# Dataframe final
final_df <- cbind(valores_reais, valores_previstos)
View(final_df)

################################################################################################################################################
                              # Segunda versão do modelo SVM - Versão com Kernel Linear e GridSearch

                              # Vamos fazer uma pesquisa em grade (Grid Search) para o ajuste de hiperparâmetros e usar Kernel linear. 
                              # Mas aqui não manteremos o custo superior a 2, para que valores discrepantes não afetem extensivamente 
                              # a criação de limites de decisão e, portanto, levem ao ajuste excessivo (overfitting).
################################################################################################################################################

set.seed(182)
?tune
modelo_grid1 <- tune(svm, 
                     Customer_cat ~ ., 
                     data = dados_treino, 
                     kernel = 'linear',
                     ranges = list(cost = c(0.05, 0.1, 0.5, 1, 2))) 


summary(modelo_grid1)

# Parâmetros do melhor modelo
modelo_grid1$best.parameters

# Melhor modelo
modelo_grid1$best.model 
modelo_v2 <- modelo_grid1$best.model 
summary(modelo_v2)

# Previsões
previsoes_v2 <- predict(modelo_v2, dados_teste)

# Matriz de Confusão e Score AUC
confusionMatrix(previsoes_v2, dados_teste$Customer_cat)
curva_roc_v2 <- multiclass.roc(response = dados_teste$Customer_cat, predictor = as.numeric(as.factor(previsoes_v2)))
curva_roc_v2$auc

# Está oferecendo um desempenho muito melhor em termos de sensibilidade na previsão de classes minoritárias

################################################################################################################################################
                    # Terceira versão do modelo SVM - Versão com Kernel RBF e Otimização no Parâmetro Gamma

                    # Vamos fazer uma pesquisa em grade para o ajuste de parâmetros com kernel radial, e não manteremos 
                    # o custo superior a 2, para que os discrepantes não afetem extensivamente a criação de limites 
                    # de decisão e, portanto, levem ao ajuste excessivo.

                    # Da mesma forma, não manteremos um valor muito abaixo de 0,001 para gamma, pois isso levaria a um excesso de ajuste.
################################################################################################################################################

set.seed(182)
modelo_grid2 <- tune(svm,
                     Customer_cat ~ .,
                     data = dados_treino,
                     kernel='radial',
                     ranges = list(cost = c(0.01,0.05,0.1,0.5,1,2),
                                   gamma = c(0.0001,0.001,0.01,.05,0.1,0.01,1,2)))

summary(modelo_grid2)

# Parâmetros do melhor modelo
modelo_grid2$best.parameters

# Melhor modelo
modelo_v3 <- modelo_grid2$best.model 

# Previsões
previsoes_v3 <- predict(modelo_v3, dados_teste)

# Matriz de Confusão e Score AUC
confusionMatrix(previsoes_v3, dados_teste$Customer_cat)
curva_roc <- multiclass.roc(response = dados_teste$Customer_cat, predictor = as.numeric(as.factor(previsoes_v3)))
curva_roc$auc

################################################################################################################################################
                    # Quarta versão do modelo SVM - Versão com Kernel Polinomial (Também calcula distância entre grupos de dados)

                    # Vamos fazer uma pesquisa em grade para ajustar parâmetros com kernel poliinomial, e não manteremos 
                    # o custo superior a 2, para que os discrepantes não afetem extensivamente a criação de limites 
                    # de decisão e, portanto, levem a um excesso de ajuste.

                    # Da mesma forma, não manteremos o grau polinomial de ordem superior a 4, pois isso levaria a um ajuste excessivo
################################################################################################################################################

set.seed(182)
modelo_grid3 <- tune(svm,
                     Customer_cat ~ .,
                     data = dados_treino,
                     kernel = 'polynomial',
                     ranges = list(cost = c(1,2), degree = c(2,3,4)))

summary(modelo_grid3)

# Parâmetros do melhor modelo
modelo_grid3$best.parameters

# Melhor modelo
modelo_v4 <- modelo_grid3$best.model 

# Previsões
previsoes_v4 <- predict(modelo_v4, dados_teste)

# Matriz de Confusão e Score AUC
confusionMatrix(previsoes_v4, dados_teste$Customer_cat)
curva_roc <- multiclass.roc(response = dados_teste$Customer_cat, predictor = as.numeric(as.factor(previsoes_v4)))
curva_roc$auc

# Podemos ver que o modelo ajustado com kernel polinomial tem uma sensibilidade fraca na previsão de clientes com alto gasto para o conjunto de testes.

################################################################################################################################################
                                                    # Comparação dos Modelos
################################################################################################################################################

# Resultados do Modelo 1
resultados_v1 <- caret::confusionMatrix(previsoes_v1, dados_teste$Customer_cat)
resultados_v1$overall
resultados_v1$byClass

# Medidas Globais do Modelo 1
acuracia_v1 <- resultados_v1$overall['Accuracy']
curva_roc <- multiclass.roc(response = dados_teste$Customer_cat, predictor = as.numeric(as.factor(previsoes_v1)))
score_auc_v1 <- curva_roc$auc

# Caso você queira outras medidas como precision e recall, lembre-se que elas são por classe
#precision_v1_classe1 <- resultados_v1$byClass[1, 'Precision']   
#precision_v1_classe2 <- resultados_v1$byClass[2, 'Precision']  
#recall_v1_classe3 <- resultados_v1$byClass[3, 'Sensitivity']

# Vetor com os resultados de avaliação do Modelo v1
vetor_modelo1 <- c("Modelo1 Kernel RBF", round(acuracia_v1, 4), round(score_auc_v1, 4))


# Medidas Globais do Modelo 2
resultados_v2 <- caret::confusionMatrix(previsoes_v2, dados_teste$Customer_cat)
acuracia_v2 <- resultados_v2$overall['Accuracy']
curva_roc <- multiclass.roc(response = dados_teste$Customer_cat, predictor = as.numeric(as.factor(previsoes_v2)))
score_auc_v2 <- curva_roc$auc

# Vetor com os resultados de avaliação do Modelo v2
vetor_modelo2 <- c("Modelo2 Kernel Linear", round(acuracia_v2, 4), round(score_auc_v2, 4))


# Medidas Globais do Modelo 3
resultados_v3 <- caret::confusionMatrix(previsoes_v3, dados_teste$Customer_cat)
acuracia_v3 <- resultados_v3$overall['Accuracy']
curva_roc <- multiclass.roc(response = dados_teste$Customer_cat, predictor = as.numeric(as.factor(previsoes_v3)))
score_auc_v3 <- curva_roc$auc

# Vetor com os resultados de avaliação do Modelo v1
vetor_modelo3 <- c("Modelo3 Kernel RBF Tunning", round(acuracia_v3, 4), round(score_auc_v3, 4))


# Medidas Globais do Modelo 4
resultados_v4 <- caret::confusionMatrix(previsoes_v4, dados_teste$Customer_cat)
acuracia_v4 <- resultados_v4$overall['Accuracy']
curva_roc <- multiclass.roc(response = dados_teste$Customer_cat, predictor = as.numeric(as.factor(previsoes_v4)))
score_auc_v4 <- curva_roc$auc

# Vetor com os resultados de avaliação do Modelo v1
vetor_modelo4 <- c("Modelo4 Kernel Polinomial", round(acuracia_v4, 4), round(score_auc_v4, 4))


# Concatenando os resultados
# Dataframe para os resultados dos modelos
?base::rbind
compara_modelos <- rbind(vetor_modelo1, vetor_modelo2, vetor_modelo3, vetor_modelo4)
View(compara_modelos)
rownames(compara_modelos) <- c("1", "2", "3", "4")
colnames(compara_modelos) <- c("Modelo", "Acuracia", "AUC")
View(compara_modelos)
class(compara_modelos)

# ggplot2 exige que os formato seja dataframe
compara_modelos <- as.data.frame(compara_modelos)
class(compara_modelos)
View(compara_modelos)

# Plot
library(ggplot2)

# Acurácia
ggplot(compara_modelos, aes(x = Modelo, y = Acuracia, fill = Modelo)) + 
  geom_bar(stat = "identity") 

# AUC
ggplot(compara_modelos, aes(x = Modelo, y = AUC, fill = Modelo)) + 
  geom_bar(stat = "identity")

# Assim, o método final proposto é baseado no Kernel Linear, a versão 2 do nosso modelo.

################################################################################################################################################
                                              # Previsões com o Modelo Escolhido
################################################################################################################################################

# Salvando o modelo selecionado
?saveRDS
saveRDS(modelo_v2, "modelos/modelo_v2.rds")

# Carregando o modelo salvo
modelo_svm <- readRDS("modelos/modelo_v2.rds")
print(modelo_svm)

# Carrega o arquivo com dados de novos clientes.
# Para esses clientes não temos a variável target, pois isso é o que queremos prever.
novos_clientes <- read.csv("dados/novos_clientes.csv", header = TRUE)
View(novos_clientes)
dim(novos_clientes)

# Fazendo previsões
previsoes_novos_clientes <- predict(modelo_svm, novos_clientes)

# Apresentando o resultado final

# Previsões
previsoes_gastos_novos_clientes <- data.frame(as.numeric(as.factor(previsoes_novos_clientes)))
colnames(previsoes_gastos_novos_clientes) <- ("Previsao_Gasto")

# Dataframe final
resultado_final <- cbind(novos_clientes, previsoes_gastos_novos_clientes)
View(resultado_final)

# Ajusta o label da previsão
library(plyr)
?mapvalues
resultado_final$Previsao_Gasto <- mapvalues(resultado_final$Previsao_Gasto,
                                                 from = c(1,2,3),
                                                 to = c("Alto", "Médio", "Baixo"))

View(resultado_final)
write.csv(resultado_final, "dados/resultado_final.csv")


