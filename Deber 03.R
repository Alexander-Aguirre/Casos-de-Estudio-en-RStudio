                             ########################################################
                             # Deber 03 - Caso de Estudio - ANN y SVM Clasificación #
                             ########################################################                                       

######################################
# Instalacion y llamado de librerias #
######################################

# Instalacion de librerias:
install.packages("readr")
install.packages("caret")
install.packages("hplot")
install.packages("car")
install.packages("cluster")
install.packages("clValid")
install.packages("aricode")
install.packages('neuralnet')
install.packages("DMwR")
install.packages("carData")

# Llamado de librerias:
library(readr)
library(corrplot)
library(caTools) #Permite hacer el "splitRatio"
library(ggplot2)
library(lattice)
library(caret) #Permite hacer el "scatterplotMatrix"
library(car)
library(dendextend)
library(cluster)
library(clValid)
library(aricode)
library(plyr)
library(neuralnet)
library(DMwR)
library(Metrics)
library(neuralnet)
library(DMwR)
library(e1071)


###########################
# Importacion del Dataset #
###########################

tiempo1 <- proc.time() # Tiempo de ejecución del aprendizaje no supervisado

train <- read_csv("C:/Users/DELL/Desktop/train.csv", col_names = TRUE)
View(train)

                                  ##############################
                                  # Aprendizaje no supervisado #
                                  ##############################





###################################
# Seleccion de Variables (Manual) #
###################################

train1 <- train[,-c(1,2,5,19,23,24,25,26,27,28,29),drop=FALSE]
View(train1)

ground = as.factor(train$Failure)


#Muestrear (aleatorio simple) y Graficar
#####set.seed(2)
split = sample.split(train1$Failure, SplitRatio = 0.7) # 70% de entrenamiento y 30% test
summary(split)
sam = subset(train1, split == TRUE)
pairs(sam)

# IMAGEN 001 - (Cuadros a blanco y negro)

c = cor(sam)
corrplot(c)

# IMAGEN 002 - (Grafico de calor de correlaciones)

scatterplotMatrix(sam) #

# IMAGEN 003 - Grafico de matriz de correlaciones

# Muchas variables no dice nada

#################################
# Escalamiento Multidimensional #
#################################

# Matriz de distancias
d = dist(train1, method = "euclidean")

# Escalamiento miltidimencional a dos dimensiones (k=2), necesario en el caso que necesite graficar.
fit = cmdscale(d,eig=TRUE, k=2) # k es el numero de dimensiones
x = fit$points[,1] 
y = fit$points[,2]
plot(x,y,col=c("blue","red")[ground], main = "Fallas Original")


# IMAGEN 01 - Representacion grafica de las fallas presentadas en el Dataset

text(x, y, labels = row.names(train), cex=1)

# IMAGEN 02 - Representacion grafica de las fallas presentadas en el Dataset con etiquetas de las instancias

####################
# Algoritmo K-Means#
####################


# Elbow

# Crea diferentes valores de k
wi = c()
for (i in 1:10) 
{
  g = kmeans(train1,i) 
  wi[i] = g$tot.withinss
}
plot((1:length(wi)),wi, xlab="Numero de Clusters", ylab="Suma Cuadrados Internos", pch=19, col="red", type = "b")

# IMAGEN 03 - Representacion grafica del algoritmo Elbow
# los grupos ideales esta en el codo, y el mismo me indica dos grupos.

grupos = kmeans(train1,2)
g1 = grupos$cluster
g2 = grupos$size
plot(x,y,col=c("Blue","green2")[g1], main = "Fallas K-Means")

# IMAGEN 04 - Grafica K-Means

############################
# Algoritmo Jerarquico DHC #
############################

hc = hclust(d, method = "complete" )
clus3 = cutree(hc, 2)
dend = as.dendrogram(hc)
dend = color_branches(dend, 2)
colors = c("cian", "green3")
plot(dend, fill = colors[clus3], cex = 0.1 , main = "Fallas DHC")


# IMAGEN 05 - Dendograma o clusteinr Jerargico

######################
# Validacion Interna #
######################

##################
# Indice de Dunn #
##################

du1 = dunn(d,g1)
du1 
du2 = dunn(d,clus3)
du2

# du1=0.1375213
# du2=0.1681254

##########################
# Coeficiente de Silueta #
##########################

sil1 = silhouette(g1,d)
plot(sil1,col=1:2, border=NA)

# IMAGEN 06 - Coeficiente de silueta de K-Means

sil2 = silhouette(clus3,d)
plot(sil2,col=4:5, border=NA)

# IMAGEN 07 - Coeficiente de silueta de DHC

# Al ser una compactacion de clustering promedio de 0.7, describe que es una muy buena compactacion
# de instancias.
######################
# Validacion Externa #
######################

###############
# ARI-AMI-NMI #
###############

# Algoritmo K-Means

ARI1= ARI(ground,g1)
ARI1
AMI1= AMI(ground,g1)
AMI1
NMI1= NMI(ground,g1,variant = c("joint"))
NMI1

# Algoritmo DHC

ARI2= ARI(ground,clus3)
ARI2
AMI2= AMI(ground,clus3)
AMI2
NMI2= NMI(ground,clus3,variant = c("joint"))
NMI2

# ARI1 = -1.160343e-05
# AMI1 = 8.135502e-06 
# NMI1 = 7.551255e-06

# ARI2 = 0.0002660148
# AMI2 = 2.864666e-06
# NMI2 = 2.639932e-06 



proc.time() - tiempo1
#  user  system elapsed 
#830.60   44.03  877.88 

# user: tiempo de la CPU dedicado a la ejecucion de las instrucciones del proceso [segundos].
# system: tiempo de la CPU empleado por el sistema operativo [segundos].
# elapsed: tiempo transcurrido real desde que se inicio el proceso [segundos].


###########################
# Aprendizaje supervisado #
###########################

###############################
# Red Neuronal Artificial ANN #
###############################

# Division el Dataset en "Training" y "Test".
# Training -> Saco el modelo (75%).
# Test     -> Pruebo que tan bueno es el modelo (25%).


tiempo2 <- proc.time()  # Tiempo de ejecución de la red Neuronal

set.seed(2)
ind_test = sample(nrow(sam), nrow(sam)/4) # 4 -> 100%/4=25%
Test = sam[ind_test, ]
Train = sam[-ind_test, ]

# Renombrar el nombre de las columnas

# x1   -> Temperatura
# x2  -> Humedad
# x3  -> Medida 1
# x4  -> Medida 2
# x5  -> Medida 3
# x6  -> Medida 4
# x7  -> Medida 5
# x8  -> Medida 6
# x9  -> Medida 7
# x10 -> Medida 8
# x11 -> Medida 9
# x12 -> Medida 10
# x13 -> Medida 11
# x14 -> Medida 12
# x15 -> Medida 13
# x16 -> Medida 14
# x17 -> Medida 15
# y   -> Fialure

colnames(Train) = c("x1","x2","x3","x4","x5","x6","x7","x8","x9","x10","x11","x12","x13","x14","x15","x16","x17","y")
Train
colnames(Test) = c("x1","x2","x3","x4","x5","x6","x7","x8","x9","x10","x11","x12","x13","x14","x15","x16","x17","y")
Test

# Entrenamiento Red Neuronal

ann = neuralnet(y ~ ., Train, hidden = c (3))
plot(ann, rep = "best")


# IMAGEN 8 - Representacion grafica de la red neuronal

# La imagen describe:
# 17 neuronas de entrada en la capa de entrada.
# 3 neurona en la capa oculta.
# 2 neuronas en la capa de salida

# Test Red Neuronal, calculo de la salida

output = compute(ann, Test[ , c("x1","x2","x3","x4","x5","x6","x7","x8","x9","x10","x11","x12","x13","x14","x15","x16","x17")])

# Predicciones

result = data.frame(  Real = Test$y, 
                      Predicted = output$net.result)

################################
# Rendimiento del clasificador #
################################

# Matriz confusion

Test2 = as.factor(Test$y)
cm = table(Test2,result$Real)
cm

# Test2   No  Yes
#   No  1372    0
#   Yes    0   11


# Evaluacion Rendimiento Clasificacion

con = confusionMatrix(Test2,result$Real)
con

#               Accuracy : 1          
#                 95% CI : (0.9973, 1)
#    No Information Rate : 0.992      
#    P-Value [Acc > NIR] : 1.598e-05  

#                  Kappa : 1          

# Mcnemar's Test P-Value : NA         
                                     
#            Sensitivity : 1.000      
#            Specificity : 1.000      
#         Pos Pred Value : 1.000      
#         Neg Pred Value : 1.000      
#             Prevalence : 0.992      
#         Detection Rate : 0.992      
#   Detection Prevalence : 0.992      
#      Balanced Accuracy : 1.000      
                                     
#       'Positive' Class : No       

accuracy.1 = con$overall[1]  #  Accuracy  :1 (Resumen general del primer elemento)
precision.1 = con$byClass[5] # Precision  :1
recall.1 = con$byClass[6]    # Recall     :1 
f1.1 = con$byClass[7]        # F1         :1


proc.time() - tiempo2

# user  system elapsed 
# 6.38    0.30    7.48 

# user: tiempo de la CPU dedicado a la ejecucion de las instrucciones del proceso [segundos].
# system: tiempo de la CPU empleado por el sistema operativo [segundos].
# elapsed: tiempo transcurrido real desde que se inicio el proceso [segundos].



########
# NOTA #
########

# Se tiene mejores resultados con el algoritmo de Clasificacion que el de Clustering,
# porque en el clustering me hizo dos grupos grandes con clases super desbalanceadas;
# un grupo pequeno de fallos y el resto (cantidad grande) de no fallos.

#################################
# Support Vector Machines - SVM #  
#################################

tiempo3 <- proc.time()  # Tiempo de ejecución de la maquina de soporte vectorial

train <- read_csv("C:/Users/DELL/Desktop/train.csv", col_names = TRUE)
View(train)

tiempo1 <- proc.time() # Tiempo de ejecución del aprendizaje no supervisado


train <- train[,-c(1,2,5,19,23,24,25,26,27,28,29),drop=FALSE]
View(train)



ground = as.factor(train$Failure)


# Validacion cruzada, Random Cross Validation
set.seed(2) # Fijar la semilla para tener 1 solo valor
split = sample.split(train$Failure, SplitRatio = 0.7) # 70% de entrenamiento y 30% test
summary(split)
sam = subset(train, split == TRUE)

# Training
training_set = subset(train, split == TRUE) # "train" debe ser del dataset original

# Test
test_set = subset(train, split == FALSE)

# Escalar a otra dimension, se hace unicamente en las variables numericas
training_set[,1:17] = scale(training_set[,1:17])
test_set[,1:17] = scale(test_set[,1:17])

# Clasificador

classifier = svm(formula = Failure~., data = training_set, 
                 type = 'C-classification', kernel = 'linear')
classifier

# Predicciones del clasificador
test_pred = predict(classifier, type = 'response', 
                    newdata = test_set[-18]) # Respuesta en relacion a la columna "Fialure"
test_pred


################################
# Rendimiento del clasificador #
################################

# Matriz confusion

test_set2 = as.factor(test_set$Failure) # <- test_set[,18]
cm = table(test_set2,test_pred) 
cm

#          test_pred
#test_set2   No  Yes
#       No  2347    2
#       Yes    9   14


# Evaluacion Rendimiento Clasificacion

con = confusionMatrix(test_pred,test_set2)
con

#               Accuracy : 0.9954          
#                 95% CI : (0.9917, 0.9977)
#    No Information Rate : 0.9903          
#    P-Value [Acc > NIR] : 0.004286        

#                  Kappa : 0.7157          

# Mcnemar's Test P-Value : 0.070440        

#            Sensitivity : 0.9991          
#            Specificity : 0.6087          
#         Pos Pred Value : 0.9962          
#         Neg Pred Value : 0.8750          
#             Prevalence : 0.9903          
#         Detection Rate : 0.9895          
#   Detection Prevalence : 0.9933          
#      Balanced Accuracy : 0.8039          

#       'Positive' Class : No      

accuracy.2 = con$overall[1]  #  Accuracy  :0.9953626 (Resumen general del primer elemento)
precision.2 = con$byClass[5] # Precision  :0.99618
recall.2 = con$byClass[6]    # Recall     :0.9991486 
f1.2 = con$byClass[7]        # F1         :0.9976621


########
# NOTA #
########

# Las metricas de la matriz de confusion son solo para clasificacion, no para regresion.

proc.time() - tSVM
#user  system elapsed 
#0.47    0.03    1.23 

# user: tiempo de la CPU dedicado a la ejecucion de las instrucciones del proceso [segundos].
# system: tiempo de la CPU empleado por el sistema operativo [segundos].
# elapsed: tiempo transcurrido real desde que se inicio el proceso [segundos].



