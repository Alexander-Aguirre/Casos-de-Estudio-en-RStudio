                             ######################################
                             #Deber 01- Caso de Estudio Clustering#
                             ######################################

# Autores:
# Alexander Aguirre
# Gabriela Gallegos
# Maribel Torres

########################################
#Clustening para el control de procesos#
########################################

#Importar Data set
install.packages("readr")
library(readr)
segmentation <- read_csv("C:/Users/DELL/Desktop/segmentation_data(M).csv", col_names = FALSE)
View(segmentation)
segmentation = data.frame(segmentation)
segmentation

# Designacion de variables
# X1 -> Cliente
# X2 -> Frequencia
# X3 -> Actualidad
# X4 -> Monetario

###############################
#Escalamiento Multidimensional#
###############################

# Matriz de distancia
d = dist(segmentation[,1:3], method = "euclidean")
d

# NOTA: Metodo para medir la distancia es Euclideano (sistema metrico en la toma de datos).

# Matriz de correlaciones

c=cor(segmentation[,1:3])
c
install.packages("corrplot")
library(corrplot)
corrplot(c)

#FIGURA 01

# Escalamiento a dos dimensiones
fit = cmdscale(d,eig=TRUE, k=2)
fit
x = fit$points[,1] 
y = fit$points[,2]
plot(x,y)

#FIGURA 02

text(x, y, labels = row.names(segmentation), cex=1) #Poner textos originales de las etiquetas.

#FIGURA 03

# Identificacion de las Clases por cada color
# Pintar la instancia en funcion de la clase (caracteristicas de la clase)

clase = segmentation[,4] # 4 es el numero de la columna donde se encuentre la variable
clase
plot(x,y,col=c(1:4)[clase], main = "segmentation Dataset Original")
.

#########################
#Algoritmo de Clustering#
#########################

################################
#Algoritmo particional, K-Means#
################################

grupos = kmeans(segmentation[,1:3],4) 
grupos
g1 = grupos$cluster
g1
g2 = grupos$size
g2
plot(x,y,col=c("red","green3","blue","aquamarine3")[g1], main = "Segmentation Dataset K-Means")

#FIGURA 04

######################################################################
#Grupos Jerarquicos, Algoritmo DHC (Divisive Hierarchical Clustering)#
######################################################################

# Parto de un solo grupo y a la final hago "n" grupos.

library("dendextend")
hc = hclust(d, method = "complete" ) 
hc
# Recibe como entrada la matriz de distancia.
# 4847 de objetos.

clus3 = cutree(hc, 4)
clus3
# Se establece que se haga "4" grupos en el dendograma.

dend = as.dendrogram(hc) # Grafica de dendograma.
dend
dend = color_branches(dend, 4) # Pintar las ramas en funcion de colores.
dend
colors = c("red", "green3", "blue","aquamarine3") # Colores que se desean poner en funcion al numero de grupos.
plot(dend, fill = colors[clus3], cex = 0.1 , main = "Clustering Jerarquico")

#FIGURA 05

#########################################
#Evaluacion de rendimiento en Clustering#
#########################################

####################
#Algoritmo de Elbow# 
####################

# Objetivos:
# Minimizar la distancia intra-cluster entre los elementos de un mismo grupo.
# Maximiza la distancia extra-cluster entre grupos de diferentes elementos.
# Crea diferentes valores de "k-grupos" ideales para trabajar.
# Nota: Se aplicara este metodo para el "Algoritmo particional, K-Means".
# tot.withinss-> Es la distancia promedio total entre custers.

wi = c()
for (i in 1:10) #Variacion de "k-grupos" desde 1 hasta 10
{
  g = kmeans(segmentation[,1:3],i) 
  wi[i] = g$tot.withinss 
}
plot((1:length(wi)),wi, xlab="Numero de Clusters", ylab="SSE: Suma Cuadrados Internos", pch=19, col="red", type = "b")
# Plot de "x" (1, 2, ..., 10)en funcion de "y"
# x-> (1:length(wi))
# y-> wi

# Nota:
# En el codo es donde se encuentra el numero ideal de grupos, lejos del mismo los grupos
# son artificiales, en este caso sera el numero de grupos ideal de 2 a 3, sin embargo al 
# hacer 4 grupos, el mismo tiende a ser artificial.

#FIGURA 06

##############################
#Metodo de Validacion Interna#
##############################

# Valida los resultados obtenidos, que tan compactados estan los grupos, que tan buenos o
# compactos son los resultados.

################
#Indice de Dunn#
################

install.packages("cluster")
install.packages("clValid")
library(cluster)
library(clValid)

du1 = dunn(d,g1) 
du1
du2 = dunn(d,clus3)
du2


# Nota:
# La mejor agrupacion de los algoritmos, es el algoritmo jerargico DHC, ya que al ser 0.005752465
# es mayor el k-means que dio 0.002523106.

########################
#Coeficiente de Silueta#
########################

# [-1,1] -> mientras mas alto es el valor, mejor rendimiento de agrupamiento.
# El coeficiente suele ser alto para grupos convexos, bien separados y con densidad alta.

sil1 = silhouette(g1,d) # Silueta N
sil1
plot(sil1,col=1:4, border=NA)

#El coeficiente promedio es de 0.55

#FIGURA 07

sil2 = silhouette(clus3,d)
sil2
plot(sil2,col=5:8, border=NA)

#El coeficiente promedio es de 0.49

#FIGURA 08

##############################
#Metodo de Validacion Externa#
##############################

install.packages("aricode")
library(aricode)
library(plyr)

##########################
#ARI, Adjusted Rand Index#
##########################

# -1<=ARI<=1, >ARI, mayor sera la semejanza entre los resultado y el "ground truth".


segmentation2<-cbind(segmentation,g1)
View(segmentation2)

ground = as.factor(segmentation2[,5])
ground

ground = revalue(ground, c("4"="VIP","3"="Nuevos","2"="VIP Potencial","1"="Baja Frecuencia"))
ground


ARI1= ARI(ground,g1)
ARI1
# AR1=1, el resultado del agrupamiento por algoritmo k-means porque agrupa mismos datos, no existe diferencia.

ARI2= ARI(ground,clus3)
ARI2
# cluss3: el resultado del agrupamiento por algoritmo jerargico DHC.

# Nota:
# El ARI2 (0.7151374) indica que tiene un 71,51%  de semejanza entre los resultado y el "ground truth".

##################################
#AMI, Adjusted Mutual Information#
##################################

# -1<=AMI<=1, 1: Coincidencia perfecta, -1:No existe coincidencia alguna.

AMI1= AMI(ground,clus3)
AMI1

# Nota:
# El AMI1 (0.7662911) posee 76,63% de coincidencia.

#####################################
#NMI, Normalized Muatual Information#
#####################################

NMI2= NMI(ground,clus3,variant = c("joint"))
NMI2

# Nota:
# El NMI2 (0.6380167) posee 63.80% de informacion mutua o comparten.
