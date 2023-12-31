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
# Importation des donn�es
df <- read.csv("datataw.csv", sep = ",", header = TRUE)
View(df)

# Messinge data
res.aggr <- aggr(df)
res.aggr$missings

# Supprimer les donn�es constantes
colonnes_constantes <- names(df)[sapply(df, function(col) length(unique(col)) == 1)]
data1 <- df[, sapply(df, function(col) length(unique(col)) > 1)]
dim(data1)
# D�terminer la taille de l'�chantillon d�sir�e pour chaque classe
sample_size <- 1000

# Appliquer le sous-�chantillonnage et le sur-�chantillonnage
ros_object <- ovun.sample(Bankrupt. ~ ., data = data1, method = "both", N = sample_size, seed = 42)

# Extraire les donn�es �quilibr�es du r�sultat
balanced_data <- ros_object$data
dim(balanced_data)


# V�rifier la distribution des classes apr�s �quilibrage
T<-table(balanced_data$Bankrupt.)
barplot(T)



# S�paration des donn�es (60%/40%)
x <- balanced_data[, -1]
dim(x)
y <- balanced_data[, 1]
length(y)

set.seed(42)
train_proportion <- 0.6
indices <- sample(1:nrow(balanced_data), size = round(train_proportion * nrow(balanced_data)))

# Diviser les donn�es en ensembles d'entra�nement et de test
train_data <- balanced_data[indices, ]
dim(train_data)
test_data <- balanced_data[-indices, ]
dim(test_data)

# Utiliser les indices pour diviser x et y
x_train_i <- x[indices, ]
dim(x_train_i)
y_train_i <- y[indices]
length(y_train_i)
x_test_i <- x[-indices, ]
dim(x_test_i)
y_test_i <- y[-indices]
length(y_test_i)


#************************************************* Analyse en composante principale******************************************
data_pca <- train_data[, -1]

res.pca <- PCA(data_pca, scale.unit = TRUE)
summary(res.pca)
variables_coord <- res.pca$var$coord
ind_coord <- res.pca$ind$coord

# Un data frame avec que les composantes principales de PCA
pca_data_c <- as.data.frame(variables_coord)

eigenvalues <- res.pca$eig
components_to_keep <- sum(eigenvalues > 1)

# Visualiser les valeurs propres et la proportion de variance expliqu�e par chaque composante principale.
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

# Cr�er un dataframe avec les composantes principales et la variable cible
y_train <- train_data$Bankrupt.
new_data_acp <- cbind(pca_data2, y_train)
new_data_acp <- as.data.frame(new_data_acp)



#********************************* Entra�ner le mod�le logistique sur les composantes principales*****************************
x <- new_data_acp[, -11]
dim(x)
y <- new_data_acp[, 11]
length(y)

model_logistic <- glm(y ~ ., data = new_data_acp)
summary(model_logistic)
print(model_logistic$aic)

# Faire des pr�dictions avec le mod�le
predicted_probabilities <- predict(model_logistic, newdata = x, type = "response")
pred <- ifelse(predicted_probabilities > 0.5, 1, 0)
length(pred)

# Matrice de confusion
conf_matrix <- confusionMatrix(as.factor(pred), as.factor(new_data_acp$y))
t<-table(as.factor(pred), as.factor(new_data_acp$y))
print(t)
accuracy_log <- sum(diag(t)) / sum(t)
cat("Accuracy log:", accuracy_log, "\n")
conf_matrix$byClass['Sensitivity']

#********************************* Sparse PLS (Partial Least Squares)******************************
x_train_i<-as.matrix(x_train_i)

sparse_pls_model <- spls(x_train_i, y_train_i, ncomp = 3, keepX = rep(10, 3))

res <- summary(sparse_pls_model)
predictions_pls <- predict(sparse_pls_model, newdata = as.matrix(x_test_i), type = "response")
predicted_values <- predictions_pls[["predict"]][, 1, 1]

# Convertir en vecteur si n�cessaire
predicted <- as.vector(predicted_values)

# Extraire les valeurs pr�dites
predictions <- ifelse(predicted > 0.5, 1, 0)
# Matrice de confusion
conf_matrix <- confusionMatrix(as.factor(predictions), as.factor(y_test_i))

# Cr�ation d'une table de contingence
t <- table(as.factor(predictions), as.factor(y_test_i))

# Calcul de l'accuracy
accuracy_pls <- sum(diag(t)) / sum(t)
cat("Accuracy pls:", accuracy_pls, "\n")


# Calculer l'erreur quadratique moyenne (RMSE)
rmse <- sqrt(mean((y_test_i - predictions)^2))


# Calculer le coefficient de d�termination (R2)
r2 <- 1 - sum((y_test_i - predictions)^2) / sum((y_test_i - mean(y_test_i))^2)

# Afficher les r�sultats
cat("RMSE:", rmse, "\n")
cat("R2:", r2, "\n")



#**************R�gression Ridge(sur les donn�es initiale )**********************
# Appliquer la r�gression ridge
x_train<-as.data.frame(x_train_i)
ridge_model <- lm.ridge(y_train_i ~ ., data = x_train, lambda = 1)
print(ridge_model$coef)
print(ridge_model$scales)

#on utilise cross validation pour determiner lambda
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

#Fitter le mod�le 
Ridge.model<-glmnet(x_train, y_train_i,alpha=0,family="binomial",lambda=cv.ridge$lambda.min)

#prediction sur la test
x_test<-as.matrix(x_test_i)
Ridge.prob<- predict(Ridge.model,newx=x_test)
Ridge.pred <- ifelse(Ridge.prob > 0.5, 1, 0)

# matrice de confusion
conf_ridge<-confusionMatrix(as.factor(Ridge.pred), as.factor(t(y_test_i)))
conf_ridge$byClass['Sensitivity']
t<-table(Ridge.pred, t(y_test_i))
accuracy_ridge_1 <- sum(diag(t)) / sum(t)
cat("Accuracy_ridge_1:", accuracy_ridge_1, "\n")


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
# Appliquer la r�gression ridge
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

#Fitter le mod�le 
Ridge.model2<-glmnet(x_train_p, y_train_p,alpha=0,family="binomial",lambda=cv.ridge2$lambda.min)

#prediction sur la test
x_test_p<-as.matrix(x_test_p)
Ridge.prob2<- predict(Ridge.model2,newx=x_test_p)
Ridge.pred2<- ifelse(Ridge.prob2 > 0.5, 1, 0)

# matrice de confusion
conf_ridge2<-confusionMatrix(as.factor(Ridge.pred2), as.factor(t(y_test_p)))
conf_ridge2$byClass['Sensitivity']#0.9285357 
tr<-table(Ridge.pred2, t(y_test_p))
accuracy_ridge_2 <- sum(diag(tr)) / sum(tr)
cat("Accuracy_ridge_2:", accuracy_ridge_2, "\n")







#**************Rgression Lasso sur les donn�es initiales**********************

#on utilise cross validation pour determiner lambda
x_train_matrix <- as.matrix(x_train_i)
y_train_matrix<-as.matrix(y_train_i)
cv.lasso <- cv.glmnet(x_train_matrix, y_train_matrix, alpha=1, family='binomial')
plot(cv.lasso)
cv.lasso$lambda.min

#l'erreur de cross validation correspond ? lambda min
cv.lasso$cvm[which(cv.lasso$lambda==cv.lasso$lambda.min)]

#Le modle Final
Lasso.model<-glmnet(x_train, y_train,alpha=1,family="binomial",lambda=cv.lasso$lambda.min)

#prediction sur la test
x_test_i<-as.matrix(x_test_i)
Lasso.prob<- predict(Lasso.model,newx=x_test_i)
Lasso.pred <- ifelse(Lasso.prob > 0.5, 1, 0)

# matrice de confusion
Y<- as.factor(t(y_test_i))
conf_lasso<-confusionMatrix(as.factor(Lasso.pred), Y)
conf_lasso$byClass['Sensitivity']

table_lasso_1<-table(Lasso.pred, t(y_test_i))
accuracy_lasso1 <- sum(diag(table_lasso_1)) / sum(table_lasso_1)
cat("Accuracy lasso 1:", accuracy_lasso1, "\n")


#mse
mean(Lasso.pred== y_test_i)

#coefficients estims par lambda min
coef(cv.lasso, cv.lasso$lambda.min)



#**************************lasso sur les composante principale ************************
#*#les donn�es 
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
cv.lasso2$lambda.min

#l'erreur de cross validation correspond ? lambda min
cv.lasso2$cvm[which(cv.lasso2$lambda==cv.lasso2$lambda.min)]

#Le modle Final
Lasso.model2<-glmnet(x_train_matrix_p, y_train_p,alpha=1,family="binomial",lambda=cv.lasso2$lambda.min)

#prediction sur la test
x_test_p<-as.matrix(x_test_p)
Lasso.prob2<- predict(Lasso.model2,x_test_p)
Lasso.pred2<- ifelse(Lasso.prob2 > 0.5, 1, 0)

# matrice de confusion
Y<- as.factor(t(y_test_p))
conf_lasso2<-confusionMatrix(as.factor(Lasso.pred2), Y)
conf_lasso2$byClass['Sensitivity']
table_lasso_2<-table(Lasso.pred2, t(y_test_p))
accuracy_lasso2 <- sum(diag(table_lasso_2)) / sum(table_lasso_2)
cat("Accuracy lasso 2:", accuracy_lasso2, "\n")


#mean
mean(Lasso.pred2== y_test_p)

#coefficients estims par lambda min
coef(cv.lasso2, cv.lasso2$lambda.min)






#******************ElasticNet sur les donn�es initiale ************************


# Ajuster le mod�le Elastic Net avec la cross-validation
x_train<-as.matrix(x_train_i)
cv.elasticnet <- cv.glmnet(x_train, y_train_i, alpha = 0.5, family = "binomial")


# S�lectionner le meilleur mod�le
best_model <- glmnet(x_train, y_train_i, alpha = 0.5, lambda = cv.elasticnet$lambda.min, family = "binomial")

# Faire des pr�dictions sur l'ensemble de test
x_test<-as.matrix(x_test_i)
probabilities <- predict(best_model, newx = x_test, s = cv.elasticnet$lambda.min, type = "response")
predictions <- ifelse(probabilities > 0.5, 1, 0)

# �valuer la performance du mod�le
confusion_matrix <- table(predictions, y_test_i)
print(confusion_matrix)
accuracy_ela_1 <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
cat("Accuracy 1:", accuracy_ela_1, "\n")


conf_ilas<-confusionMatrix(as.factor(predictions),as.factor(y_test_i) )

#*********************ElasticNet sur les composante principale*****************************
# Ajuster le mod�le Elastic Net avec la cross-validation
x_train_p<-as.matrix(x_train_p)
cv.elasticnet2 <- cv.glmnet(x_train_p, y_train_p, alpha = 0.5, family = "binomial")


# S�lectionner le meilleur mod�le
best_model2 <- glmnet(x_train_p, y_train_p, alpha = 0.5, lambda = cv.elasticnet2$lambda.min, family = "binomial")

# Faire des pr�dictions sur l'ensemble de test
x_test_p<-as.matrix(x_test_p)
probabilities2 <- predict(best_model2, newx = x_test_p, s = cv.elasticnet2$lambda.min, type = "response")
predictions2 <- ifelse(probabilities2 > 0.5, 1, 0)

# �valuer la performance du mod�le
confusion_matrix2 <- table(predictions2, y_test_p) 
print(confusion_matrix2)
accuracy_ela_2 <- sum(diag(confusion_matrix2)) / sum(confusion_matrix2)
cat("Accuracy2:", accuracy_ela_2, "\n")

conf_ilas<-confusionMatrix(as.factor(predictions2),as.factor(y_test_p) )



