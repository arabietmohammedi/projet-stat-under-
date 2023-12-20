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

# Importation des données
df <- read.csv("datataw.csv", sep = ",", header = TRUE)
View(df)

# Messinge data
res.aggr <- aggr(df)
res.aggr$missings

# Supprimer les données constantes
colonnes_constantes <- names(df)[sapply(df, function(col) length(unique(col)) == 1)]
data1 <- df[, sapply(df, function(col) length(unique(col)) > 1)]

# under simpling
under <- ovun.sample(Bankrupt. ~ ., data = data1, method = "under")
data <- under$data
dim(data)
T2 <- table(data$Bankrupt.)
barplot(T2)

# Séparation des données (60%/40%)
x <- data[, -1]
y <- data[, 1]
set.seed(42)

train_proportion <- 0.6
indices <- sample(1:nrow(data), size = round(train_proportion * nrow(data)))

# Diviser les données en ensembles d'entraînement et de test
train_data <- data[indices, ]
dim(train_data)
test_data <- data[-indices, ]
dim(test_data)

# Utiliser les indices pour diviser x et y
x_train_i <- x[indices, ]
y_train_i <- y[indices]
x_test_i <- x[-indices, ]
y_test_i <- y[-indices]


#************************************************* Analyse en composante principale******************************************
#on va fair un pca sur les données train
data_pca <- train_data[, -1]
dim(data_pca)
res.pca <- PCA(data_pca, scale.unit = TRUE)
summary(res.pca)
variables_coord <- res.pca$var$coord
ind_coord <- res.pca$ind$coord

# Un data frame avec que les composantes principales de PCA
pca_data_c <- as.data.frame(variables_coord)
eigenvalues <- res.pca$eig
components_to_keep <- sum(eigenvalues > 1)

# Visualiser les valeurs propres et la proportion de variance expliquée par chaque composante principale.
fviz_screeplot(res.pca, choice = "eigenvalue")

# Essayer avec 10 composantes principales
res.pca2 <- PCA(data_pca, scale.unit = TRUE, ncp = 10)
print(res.pca2)
summary(res.pca2)

variables_coord2 <- res.pca2$var$coord
ind_coord2 <- res.pca2$ind$coord

# Afficher les composantes principales
print(variables_coord)

# Un data frame avec que les composantes principales de PCA
pca_data2 <- as.data.frame(ind_coord2)
dim(pca_data2)

# Créer un dataframe avec les composantes principales et la variable cible
y_train <- train_data$Bankrupt.
new_data_acp <- cbind(pca_data2, y_train)
new_data_acp <- as.data.frame(new_data_acp)
dim(new_data_acp)

#********************************* Entraîner le modèle logistique sur les composantes principales*****************************
x <- new_data_acp[, -11]
y <- new_data_acp[, 11]

model_logistic <- glm(y ~ ., data = new_data_acp)
summary(model_logistic)
print(model_logistic$aic)

# Faire des prédictions avec le modèle
predicted_probabilities <- predict(model_logistic, newdata = x, type = "response")
pred <- ifelse(predicted_probabilities > 0.5, 1, 0)

# Matrice de confusion
conf_matrix <- confusionMatrix(as.factor(pred), as.factor(new_data_acp$y))
conf_matrix
t<-table(as.factor(pred), as.factor(new_data_acp$y))
accuracy1 <- sum(diag(t)) / sum(t)
cat("Accuracy2:", accuracy1, "\n")
conf_matrix$byClass['Sensitivity']

#********************************* Sparse PLS (Partial Least Squares)******************************
x_train_i<-as.matrix(x_train_i)

sparse_pls_model <- spls(x_train_i, y_train_i, ncomp = 3, keepX = rep(10, 3))

res <- summary(sparse_pls_model)
predictions_pls <- predict(sparse_pls_model, newdata = as.matrix(x_test_i), type = "response")
predicted_values <- predictions_pls[["predict"]][, 1, 1]

# Convertir en vecteur si nécessaire
predicted <- as.vector(predicted_values)

# Extraire les valeurs prédites
predictions <- ifelse(predicted > 0.5, 1, 0)

# Matrice de confusion
conf_matrix <- confusionMatrix(as.factor(predictions),as.factor( y_test_i))
conf_matrix
# Création d'une table de contingence
t <- table(as.factor(predictions), as.factor(y_test_i))

# Calcul de l'accuracy
accuracy1 <- sum(diag(t)) / sum(t)
cat("Accuracy2:", accuracy1, "\n")

# Affichage de la sensibilité (True Positive Rate)
conf_matrix$byClass['Sensitivity']


# Calculer l'erreur quadratique moyenne (RMSE)
rmse <- sqrt(mean((y_test_i - predictions)^2))

# Calculer le coefficient de détermination (R2)
r2 <- 1 - sum((y_test_i - predictions)^2) / sum((y_test_i - mean(y_test_i))^2)

# Afficher les résultats
cat("RMSE:", rmse, "\n")




#**************Régression Ridge(sur les données initiale )**********************
# Appliquer la régression ridge
x_train<-as.data.frame(x_train_i)
ridge_model <- lm.ridge(y_train_i ~ ., data = x_train, lambda = 1)
print(ridge_model$coef)
print(ridge_model$scales)

#on utilise cross validation pour determiner le  lambda optimal
cv.ridge <- cv.glmnet(x_train_i, y_train_i, alpha = 0, family = 'gaussian')
plot(cv.ridge)
#on utilise cross validation pour determiner lambda
x_train<-as.matrix(x_train_i)
cv.ridge <- cv.glmnet(x_train, y_train_i, alpha=0, family='binomial')
plot(cv.ridge)

#lambda min
cv.ridge$lambda.min

#l'erreur de cross validation correspond 
cv.ridge$cvm[which(cv.ridge$lambda==cv.ridge$lambda.min)]

#Fitter le modèle 
Ridge.model<-glmnet(x_train, y_train_i,alpha=0,family="binomial",lambda=cv.ridge$lambda.min)

#prediction sur le test
x_test<-as.matrix(x_test_i)
Ridge.prob<- predict(Ridge.model,newx=x_test)
Ridge.pred <- ifelse(Ridge.prob > 0.5, 1, 0)

# matrice de confusion
conf_ridge<-confusionMatrix(as.factor(Ridge.pred), as.factor(t(y_test_i)))
conf_ridge$byClass['Sensitivity']
table_ridge<-table(Ridge.pred, t(y_test_i))
# Calculer l'accuracy
accuracy_ridge <- sum(diag(table_ridge)) / sum(table_ridge)
cat("Accuracy Ridge:", accuracy_ridge, "\n")



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

#on utilise cross validation pour determiner lambda optimale 
x_train_p<-as.matrix(x_train_p)
cv.ridge2 <- cv.glmnet(x_train_p, y_train_p, alpha = 0, family = 'gaussian')
plot(cv.ridge2)

#on utilise cross validation pour determiner lambda
x_train_p<-as.matrix(x_train_p)
cv.ridge2 <- cv.glmnet(x_train_p, y_train_p, alpha=0, family='binomial')
plot(cv.ridge2)

#lambda min
cv.ridge2$lambda.min

#l'erreur de cross validation correspond  
cv.ridge2$cvm[which(cv.ridge2$lambda==cv.ridge2$lambda.min)]

#Fitter le modèle 
Ridge.model2<-glmnet(x_train_p, y_train_p,alpha=0,family="binomial",lambda=cv.ridge2$lambda.min)

#prediction sur la test
x_test_p<-as.matrix(x_test_p)
Ridge.prob2<- predict(Ridge.model2,newx=x_test_p)
Ridge.pred2<- ifelse(Ridge.prob2 > 0.5, 1, 0)

# matrice de confusion
conf_ridge2<-confusionMatrix(as.factor(Ridge.pred2), as.factor(t(y_test_p)))
conf_ridge2$byClass['Sensitivity']#0.9285357 
table_ridge2<-table(Ridge.pred2, t(y_test_p))
# Calculer l'accuracy
accuracy_ridge2 <- sum(diag(table_ridge2)) / sum(table_ridge2)
cat("Accuracy Ridge:", accuracy_ridge2, "\n")







#**************Rgression Lasso sur les données initiales**********************

#on utilise cross validation pour determiner lambda
x_train_matrix <- as.matrix(x_train_i)
y_train_matrix<-as.matrix(y_train_i)
cv.lasso <- cv.glmnet(x_train_matrix, y_train_matrix, alpha=1, family='binomial')
plot(cv.lasso)
cv.lasso$lambda.min

#l'erreur de cross validation correspond a lambda min
cv.lasso$cvm[which(cv.lasso$lambda==cv.lasso$lambda.min)]

#Le modle Final
Lasso.model<-glmnet(x_train_matrix, y_train_matrix,alpha=1,family="binomial",lambda=cv.lasso$lambda.min)

#prediction sur la test
x_test_i<-as.matrix(x_test_i)
Lasso.prob<- predict(Lasso.model,newx=x_test_i)
Lasso.pred <- ifelse(Lasso.prob > 0.5, 1, 0)

# matrice de confusion
Y<- as.factor(t(y_test_i))
conf_lasso<-confusionMatrix(as.factor(Lasso.pred), Y)
conf_lasso
conf_lasso$byClass['Sensitivity']

tabel1<-table(Lasso.pred, t(y_test_i))
accuracy_lasso1 <- sum(diag(tabel1)) / sum(tabel1)
cat("Accuracy lasso:", accuracy_lasso1, "\n")


#mse
mean(Lasso.pred== y_test_i)

#coefficients estims par lambda min
coef(cv.lasso, cv.lasso$lambda.min)



#**************************lasso sur les composante principale ************************
#*#les données 
set.seed(123)
indices_train <- sample(1:nrow(new_data_acp), 0.7 * nrow(new_data_acp))
train_data_pca <- new_data_acp[indices_train, ]
dim(train_data_pca)
test_data_pca <- new_data_acp[-indices_train, ]
dim(test_data_pca)
x<-new_data_acp[, -11]
dim(x)
y<-new_data_acp[,11]
length(y)
x_train_p <- x[indices_train , ]
dim(x_train_p)
y_train_p <- y[indices_train]
length(y_train_p)
x_test_p<- x[-indices_train, ]
dim(x_test_p)
y_test_p<- y[-indices_train]
length(y_test_p)

#on utilise cross validation pour determiner lambda
x_train_matrix_p <- as.matrix(x_train_p)
cv.lasso2 <- cv.glmnet(x_train_matrix_p, y_train_p, alpha=1, family='binomial')
plot(cv.lasso2)
cv.lasso2$lambda.min

#l'erreur de cross validation correspond pour  lambda min
cv.lasso2$cvm[which(cv.lasso2$lambda==cv.lasso2$lambda.min)]

#Le modle Final
Lasso.model2<-glmnet(x_train_matrix_p, y_train_p,alpha=1,family="binomial",lambda=cv.lasso2$lambda.min)

#prediction sur les données  test des composant principal
x_test_p<-as.matrix(x_test_p)
Lasso.prob2<- predict(Lasso.model2,x_test_p)
Lasso.pred2<- ifelse(Lasso.prob2 > 0.5, 1, 0)

# matrice de confusion
Y<- as.factor(t(y_test_p))
conf_lasso2<-confusionMatrix(as.factor(Lasso.pred2), Y)
conf_lasso2$byClass['Sensitivity']
tabel2<-table(Lasso.pred2, t(y_test_p))

accuracy_lasso2 <- sum(diag(tabel2)) / sum(tabel2)
cat("Accuracy lasso:", accuracy_lasso2, "\n")


#mean
mean(Lasso.pred2== y_test_p)

#coefficients estims par lambda min
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

conf_ilas<-confusionMatrix(as.factor(predictions),as.factor(y_test_i) )


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





















