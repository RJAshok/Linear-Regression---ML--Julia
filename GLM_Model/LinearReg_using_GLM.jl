#This file is used to calculate an optimal model without using Machine Learning
#It uses the ols model present in GLM module
#We do this to compare our model with this

using CSV, Plots, TypedTables, GLM

t=Table(X=X,Y=Y)

ols=lm(@formula(Y~X),t)


plot!(X,predict(ols),color=:red)