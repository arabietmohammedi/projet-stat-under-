library(missMDA)
library(missForest)
library(mice)
library(DescToolsAddIns)
library(VIM)
library(MASS)
library(FactoMineR)
library(factoextra)
library(elasticnet)
library(rgl)
library(pheatmap)
library(mixOmics)
library(ROSE)
library(smotefamily)
library(caret)
library(plspm)
library(pls)
library(glmnet)
library(logistf)

#*************************************************************************
######## importation des données 
df <- read.csv("datataw.csv", sep = ",", header = TRUE)
View(df)

###messinge data ###########
res.aggr<-aggr(df)
res.aggr$missings

####### supremer les données constante 
# Identifiez les noms des colonnes constantes
colonnes_constantes <- names(df)[sapply(df, function(col) length(unique(col)) == 1)]#"Net.Income.Flag"

# Supprimer les colonnes constantes dans un dataframe
data1<- df[,sapply(df, function(col) length(unique(col)) > 1)]

t<-table(data1$Bankrupt.)
barplot(t)

#data1 c'est les donnés sans la variable constante 

### ouver simplinge##### 
over=ovun.sample(Bankrupt.~.,data=data1,method="over")
data=over$data
T2<-table(data$Bankrupt.)
barplot(T2)

#data c'est les données apres l'ouver simplinge

# Séparation des données (60%/40%)
x <- data[, -1]
y <- data[, 1]
set.seed(42)

train_proportion <- 0.6
indices <- sample(1:nrow(data), size = round(train_proportion * nrow(data)))

# Diviser les données en ensembles d'entraînement et de test
train_data <- data[indices, ]
test_data <- data[-indices, ]

# Utiliser les indices pour diviser x et y
x_train_i <- x[indices, ]
y_train_i <- y[indices]
x_test_i <- x[-indices, ]
y_test_i <- y[-indices]









#*********************************************analyse en composante principale *******************************************

data_frame <- data

# Exclure la première colonne sans la variable target
data_pca <- data_frame[, -1]

res.pca =PCA(data_pca,scale.unit=TRUE)
summary(res.pca)
variables_coord <- res.pca$var$coord
ind_coord <- res.pca$ind$coord
#un data frem avec que les composante prensipale de pca 
pca_data_c <- as.data.frame(variables_coord)

eigenvalues <- res.pca$eig
components_to_keep <- sum(eigenvalues > 1)

#visualiser les valeurs propres et la proportion de variance expliquée par chaque composante principale.
fviz_screeplot(res.pca, choice = "eigenvalue")

#Cette ligne extrait les coordonnées des individus dans l'espace des composantes principales à 
#partir de l'objet résultant de l'ACP
c<-get_pca(res.pca, element = c("ind"))
c$coord

#Cette ligne génère un graphique de la variance expliquée par chaque composante principale.
# Cela peut aider à choisir le nombre optimal de 
#composantes principales à conserver.
plot(eigenvalues, type = "b", ylab = "Variance expliquée", xlab = "Composantes principales", main = "Variance expliquée par les composantes principales")

#######on essey avec 10 composant principale 
res.pca2 =PCA(data_pca,scale.unit=TRUE,ncp = 10)
print(res.pca2)
summary(res.pca2)
# Accéder aux coordonnées des variables
variables_coord2 <- res.pca2$var$coord
ind_coord2 <- res.pca2$ind$coord

# Afficher les composantes principales
print(variables_coord2)
#un data frem avec que les composante prensipale de pca 
pca_data2 <- as.data.frame(ind_coord2)
# Sélectionnez les premières 94 observations de la variable cible
#subset_target <- sample(data$Bankrupt., 94)
# Créez un dataframe avec les composantes principales et la variable cible
y<-data$Bankrupt.
new_data_acp<- cbind(pca_data2, y)
new_data_acp <- as.data.frame(new_data_acp)



plot.PCA(res.pca,axes=c(1,2), cex=0.7)
plot.PCA(res.pca,axes=c(1,3), cex=0.7)
plot.PCA(res.pca,axes=c(2,3), cex=0.7)
plot.PCA(res.pca,axes=c(1,4), cex=0.7)
plot.PCA(res.pca,axes=c(1,5), cex=0.7)
plot.PCA(res.pca,axes=c(2,6), cex=0.7)

#******************** entrnaire un modele sur les composante prncipale ************

# 4. Divisez les données en ensembles d'entraînement et de test
set.seed(123)
indices_train <- sample(1:nrow(new_data_acp), 0.7 * nrow(new_data_acp))
train_data_pca <- new_data_acp[indices_train, ]
test_data_pca <- new_data_acp[-indices_train, ]
x<-new_data_acp[, -11]
y<-new_data_acp[,11]
x_train <- x[indices_train , ]
y_train <- y[indices_train]
x_test <- x[-indices_train, ]
y_test <- y[-indices_train]

#Entraînez le modèle logistique sur les composante principale 
y<-train_data_pca$Bankrupt.
model_logistic <- glm(y ~ ., data = train_data_pca)
summary(model_logistic)
print(model_logistic$aic) 

#faire de la predection avec le modl 
predicted_probabilities <- predict(model_logistic, newdata = x_test, type = "response")
pred <- ifelse(predicted_probabilities > 0.5, 1, 0)

# matrice de confusion 
conf_ridge<-confusionMatrix(as.factor(pred), as.factor(t(y_test)))
conf_ridge
conf_ridge$byClass['Sensitivity']#Sensitivity 0.834083 
table(pred, t(y_test))
#pred    0    1
#0 1669  248
#1  332 1737


#****************************************Sparse PLS (Partial Least Squares) *********

# Appliquer la PLS sparse
y_train<-as.numeric(y_train_i)
x_train<-as.matrix(x_train_i)
sparse_pls_model <- spls(x_train, y_train, ncomp = 3, keepX = rep(10, 3))

res<-summary(sparse_pls_model)
predictions_pls <- predict(sparse_pls_model, newdata = x_test_i)



# Extraire les valeurs prédites
predictions <- predictions_pls[["predict"]][, 1, 1]
predictions <- ifelse(predictions > 0.5, 1, 0)

## Calculer l'erreur quadratique moyenne (RMSE)
rmse <- sqrt(mean((y_test_i - predictions)^2))#0.3796989

# Calculer le coefficient de détermination (R2)
r2 <- 1 - sum((y_test_i - predictions)^2) / sum((y_test_i - mean(y_test_i))^2)

# Afficher les résultats
cat("RMSE:", rmse, "\n")
cat("R2:", r2, "\n")





#**************Régression Ridge(sur les données initiale )**********************
# Appliquer la régression ridge
ridge_model <- lm.ridge(y_train_i ~ ., data = x_train_i, lambda = 1)
print(ridge_model$coef)
print(ridge_model$scales)

#on utilise cross validation pour determiner lambda
x_train_i<-as.matrix(x_train_i)
cv.ridge <- cv.glmnet(x_train_i, y_train_i, alpha = 0, family = 'gaussian')
plot(cv.ridge)

#on utilise cross validation pour determiner lambda
cv.ridge <- cv.glmnet(x_train_i, y_train_i, alpha=0, family='binomial')
plot(cv.ridge)

#lambda min
cv.ridge$lambda.min #0.03073555

#l'erreur de cross validation correspond 
cv.ridge$cvm[which(cv.ridge$lambda==cv.ridge$lambda.min)]

#Fitter le modèle 
Ridge.model<-glmnet(x_train_i, y_train_i,alpha=0,family="binomial",lambda=cv.ridge$lambda.min)

#prediction sur la test
x_test<-as.matrix(x_test_i)
Ridge.prob<- predict(Ridge.model,newx=x_test)
Ridge.pred <- ifelse(Ridge.prob > 0.5, 1, 0)

# matrice de confusion
conf_ridge<-confusionMatrix(as.factor(Ridge.pred), as.factor(t(y_test_i)))
conf_ridge$byClass['Sensitivity']#0.9285357 
table(Ridge.pred, t(y_test_i))


#************************Rgression ridge sur les composante  principale *********
set.seed(123)
indices_train <- sample(1:nrow(new_data_acp), 0.7 * nrow(new_data_acp))
train_data_pca <- new_data_acp[indices_train, ]
test_data_pca <- new_data_acp[-indices_train, ]
x<-new_data_acp[, -11]
y<-new_data_acp[,11]
x_train_p <- x[indices_train , ]
y_train_p<- y[indices_train]
x_test_p <- x[-indices_train, ]
y_test_p<- y[-indices_train]
# Appliquer la régression ridge
ridge_model2 <- lm.ridge(y_train_p ~ ., data = x_train_p, lambda = 1)
print(ridge_model2$coef)
print(ridge_model2$scales)

#on utilise cross validation pour determiner lambda
x_train_p<-as.matrix(x_train_p)
cv.ridge2 <- cv.glmnet(x_train_p, y_train_p, alpha = 0, family = 'gaussian')
plot(cv.ridge2)

#on utilise cross validation pour determiner lambda
x_train_p<-as.matrix(x_train_p)
cv.ridge2 <- cv.glmnet(x_train_p, y_train_p, alpha=0, family='binomial')
plot(cv.ridge2)

#lambda min
cv.ridge2$lambda.min

#l'erreur de cross validation correspond ? 
cv.ridge2$cvm[which(cv.ridge2$lambda==cv.ridge2$lambda.min)]

#Fitter le modèle 
Ridge.model2<-glmnet(x_train_p, y_train_p,alpha=0,family="binomial",lambda=cv.ridge2$lambda.min)

#prediction sur la test
x_test_p<-as.matrix(x_test_p)
Ridge.prob2<- predict(Ridge.model2,newx=x_test_p)
Ridge.pred2<- ifelse(Ridge.prob2 > 0.5, 1, 0)

# matrice de confusion
conf_ridge2<-confusionMatrix(as.factor(Ridge.pred2), as.factor(t(y_test_p)))
conf_ridge2
conf_ridge2$byClass['Sensitivity']#0.9285357 
table(Ridge.pred2, t(y_test_p))






#**************Rgression Lasso sur les données initiales**********************
#on utilise cross validation pour determiner lambda
x_train_matrix <- as.matrix(x_train_i)
y_train_matrix<-as.matrix(y_train_i)
cv.lasso <- cv.glmnet(x_train_matrix, y_train_matrix, alpha=1, family='binomial')
plot(cv.lasso)
cv.lasso$lambda.min

#l'erreur de cross validation correspond pour determiner lambda min
cv.lasso$cvm[which(cv.lasso$lambda==cv.lasso$lambda.min)]

#Le modle Final
Lasso.model<-glmnet(x_train_i, y_train_i,alpha=1,family="binomial",lambda=cv.lasso$lambda.min)

#prediction sur la test
x_test_i<-as.matrix(x_test_i)
Lasso.prob<- predict(Lasso.model,newx=x_test_i)
Lasso.pred <- ifelse(Lasso.prob > 0.5, 1, 0)
length(Lasso.pred)
Lasso.pred<-as.factor(Lasso.pred)
y_test_i<- as.factor(y_test_i)
levels(Lasso.pred) <- levels(y_test_i)
conf_lasso <- confusionMatrix(Lasso.pred, y_test_i)
conf_lasso$byClass['Sensitivity'] 

T<-table(Lasso.pred, t(y_test_i))
accuracy2 <- sum(diag(T)) / sum(T)
cat("Accuracy:", accuracy2, "\n")


#mse
mean(Lasso.pred== y_test_i)#0.8093327

#coefficients estims par lambda min
coef(cv.lasso, cv.lasso$lambda.min)



#***************************lasso sur les composante principale ************************
#*#les données 
set.seed(123)
indices_train <- sample(1:nrow(new_data_acp), 0.7 * nrow(new_data_acp))
train_data_pca <- new_data_acp[indices_train, ]
test_data_pca <- new_data_acp[-indices_train, ]
x<-new_data_acp[, -11]
y<-new_data_acp[,11]
x_train_p <- x[indices_train , ]
y_train_p <- y[indices_train]
x_test_p<- x[-indices_train, ]
y_test_p<- y[-indices_train]

#on utilise cross validation pour determiner lambda
x_train_matrix_p <- as.matrix(x_train_p)
cv.lasso2 <- cv.glmnet(x_train_matrix_p, y_train_p, alpha=1, family='binomial')
plot(cv.lasso2)
cv.lasso2$lambda.min#0.02069778

#l'erreur de cross validation correspond ? lambda min
cv.lasso2$cvm[which(cv.lasso2$lambda==cv.lasso2$lambda.min)]# 0.7659184

#Le modle Final
Lasso.model2<-glmnet(x_train_matrix_p, y_train_p,alpha=1,family="binomial",lambda=cv.lasso2$lambda.min)

#prediction sur la test
x_test_p<-as.matrix(x_test_p)
Lasso.prob2<- predict(Lasso.model2,x_test_p)
Lasso.pred2<- ifelse(Lasso.prob2 > 0.5, 1, 0)

# matrice de confusion
Y<- as.factor(t(y_test_p))
conf_lasso2<-confusionMatrix(as.factor(Lasso.pred2), Y)
conf_lasso2$byClass['Sensitivity']# 0.910045 
t<-table(Lasso.pred2, t(y_test_p))
#Lasso.pred2    0    1
# 1821  580
#1  180 1405
accuracy1 <- sum(diag(t)) / sum(t)
cat("Accuracy:", accuracy1, "\n")

#mean
mean(Lasso.pred2== y_test_p)#0.8093327

#coefficients estim?s par lambda min
coef(cv.lasso2, cv.lasso2$lambda.min)






#******************ElasticNet sur les données initiale ************************


# Ajuster le modèle Elastic Net avec la cross-validation
x_train<-as.matrix(x_train_i)
cv.elasticnet <- cv.glmnet(x_train, y_train_i, alpha = 0.5, family = "binomial")


# Sélectionner le meilleur modèle
best_model <- glmnet(x_train, y_train_i, alpha = 0.5, lambda = cv.elasticnet$lambda.min, family = "binomial")

# Faire des prédictions sur l'ensemble de test
x_test<-as.matrix(x_test_i)
probabilities <- predict(best_model, newx = x_test, s = cv.elasticnet$lambda.min, type = "response")
predictions <- ifelse(probabilities > 0.5, 1, 0)

# Évaluer la performance du modèle
confusion_matrix <- table(predictions, y_test_i)
print(confusion_matrix)
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
cat("Accuracy:", accuracy, "\n")


#*********************ElasticNet sur les composante principale*****************************
# Ajuster le modèle Elastic Net avec la cross-validation
x_train_p<-as.matrix(x_train_p)
cv.elasticnet2 <- cv.glmnet(x_train_p, y_train_p, alpha = 0.5, family = "binomial")


# Sélectionner le meilleur modèle
best_model2 <- glmnet(x_train_p, y_train_p, alpha = 0.5, lambda = cv.elasticnet2$lambda.min, family = "binomial")

# Faire des prédictions sur l'ensemble de test
x_test_p<-as.matrix(x_test_p)
probabilities2 <- predict(best_model2, newx = x_test_p, s = cv.elasticnet2$lambda.min, type = "response")
predictions2 <- ifelse(probabilities2 > 0.5, 1, 0)

# Évaluer la performance du modèle
confusion_matrix2 <- table(predictions2, y_test_p) 
print(confusion_matrix2)
accuracy2 <- sum(diag(confusion_matrix2)) / sum(confusion_matrix2)
cat("Accuracy2:", accuracy2, "\n")

conf_ilas<-confusionMatrix(as.factor(predictions2),as.factor(y_test_p) )

















 








