# Definindo o diretório de trabalho
getwd()
setwd("/Cap11_Classificacao_Regressao_Support_Vector_Machines_SVMS/R")

# Pacotes
install.packages("gains")
install.packages("pROC")
install.packages("ROSE")
install.packages("mice")
library(dplyr) # Manipulação de Dados
library(caret) # Machine Learning
library(gains) # Interpretar modelo SVM
library(pROC)  # Construir a curva ROC para interpretar as métricas
library(ROCR)  # Construir gráficos para interpretação de métricas
library(ROSE)  # Manipulação de dados
library(e1071) # Também tem o Algoritmo SVM
library(mice)  # Inputação de Dados para valores missing

# Carregando os dados
dataset_clientes <- read.csv("dados/cartoes_clientes.csv")
View(dataset_clientes)

################################################################################################################################################
                                            #### Análise Exploratória dos Dados #### 
################################################################################################################################################
str(dataset_clientes)
summary(dataset_clientes)
summary(dataset_clientes$card2spent)

# Removemos a variável com ID do cliente pois não é necessário
dataset_clientes <- dataset_clientes[-1]
View(dataset_clientes)

# Checando a existência de valores missing em todas variáveis do dataset
sapply(dataset_clientes, function(x)sum(is.na(x)))

# Checando se a variável target está balanceada
# Na etapa de treinamento os dados balanceados ajudam a tornar o modelo mais generalizado,
# Sendo capaz de aprender melhor e obter melhores resultados com novos dados.
table(dataset_clientes$Customer_cat)
prop.table(table(dataset_clientes$Customer_cat)) * 100

# Verificar os dados da variável target como um dataframe
as.data.frame(table(dataset_clientes$Customer_cat))

########################################################################################################################################################
                                                    #### Análise Visual ####
########################################################################################################################################################

# BoxPlot e Histograma (São gráficos de uma única variável)
boxplot(dataset_clientes$card2spent)
summary(dataset_clientes$card2spent)
hist(dataset_clientes$card2spent)

boxplot(dataset_clientes$hourstv)
summary(dataset_clientes$hourstv)
hist(dataset_clientes$hourstv)

# Verificando se existe relação entre as duas variáveis utilizando gráficos: Scatter Plot
# Baseado no resultado, não existe uma correlação entre gasto de cartão e horas de TV. (Um marketing em TV não seria uma boa abordagem para elevar os gastos)
plot(dataset_clientes$card2spent, dataset_clientes$hourstv, xlab = "Gasto Cartão", ylab = "Horas TV")
