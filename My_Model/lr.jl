#Andrew NG Notation
#theta_0 - y-intercept
#theta_1 - slope
#f(x) - y=mx+b (I have used f(x) instead)
#m - No. of samples
#y_hat - predicted value of y
#J - cost function
#pd in the function name stands for partial derivative
#pd_theta_0 and pd_theta_1 are used to calculate part of the partial derivative and is used in updating theta_0 and theta_1 values with the help of the learning rate 
#alpha_0 and alpha_1 are the learning rates


#Import the necessary modules
using CSV, Plots

#Import the CSV file through the use of CSV module
data=CSV.File("/home/lophius/Dev/Linear Regression/Datasets/normalized_train.csv")

#Store the features of the CSV file as a vector
X=data.x
Y=data.y

#Initialize the backend and the plot height and width
gr(size = (900,900))

#Create a scatter plot of the different points
pr_scatter = scatter(X,Y,
    xlims=(0,2),
    ylims=(0,2),
    xlabel = "X",
    ylabel = "Y",
    legend = false,
    title = "Linear Regression",
color = :blue)

#Initialize eopch value - Epoch is how long the model has been trained
epochs=0

#Initialize theta_0 and theta_1 
theta_0 = 0.0
theta_1 = 0.0

#Creating generic function
f(x)=theta_0 .+ theta_1*X


#Plotting the model, at the first iteration with theta values as 0, it will overlap with the x-axis
plot!(X,f(X),color=:blue,linedwidth=3)

#No of samples
m=length(X)

#Predicted value
y_hat = f(X)

#Cost function
function cost(X,Y)
    (1/(2*m))*sum((y_hat-Y).^2)
end

#Initialized cost function
J = cost(X,Y)

#Keeps track of the different values of the cost
J_history=[]

#Pushes the Cost function to the J_history which keeps track of it
push!(J_history,J)

#Partial derivative function for theta_0
function pd_theta_0(X,Y)
    y_hat=f(X)
    return (1/m)*sum(y_hat .- Y)
end

#Partial derivative function for theta_1
function pd_theta_1(X,Y)
    y_hat=f(X)
    return (1/m)*sum((y_hat .- Y).*X)
end

#Learning rates
alpha_0 = 0.01
alpha_1 = 0.01

#The following loop trains the model 750 times
for i in 1:100000
    theta_0-=0.01 * pd_theta_0(X,Y)
    theta_1-=0.01 * pd_theta_1(X,Y)

    y_hat = f(X)

    J=cost(X,Y)

    push!(J_history,J)

    epochs+=1

    #The below line can be uncommented to see how the model evolves for either visual purposes or debugging
    #display(plot!(X,y_hat,color=:blue,alpha=0.5, title="Model after epoch=$epochs"))
end

#Plots the model after training along with the data points
scatter(X,Y,
xlims=(0,2),
ylims=(0,2),
xlabel = "X",
ylabel = "Y",
legend = false,
title = "Linear Regression",
color = :blue)
plot!(X,y_hat,color=:green,alpha=0.5, title="Model after epoch=$epochs")


#Plots the learning curve
gr(size=(900,900))
learning_curve=plot(0:epochs,J_history,
xlabel="Epoch",
ylabel="Cost",
title="Learning Curve",
color=:blue,
linewidth=2)
