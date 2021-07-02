#
#     Description of this R script:
#     R interface for SMME routines.
#
#     Intended for use with R.
#     Copyright (C) 2021 Adam Lund
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>
#

#' @aliases SMME_predict SMME.predict
#' @title Make Prediction From a SMME Object
#'
#' @description  Given new covariate data this function computes the linear predictors
#' based on the estimated model coefficients in an object produced by the function
#' \code{softmaximin}. Note that the data can be supplied in two different
#' formats: i) as a \eqn{n' \times p} matrix (\eqn{p} is the number of model
#' coefficients and \eqn{n'} is the number of new data points) or ii) as a list
#' of two or three Kronecker component matrices each of size \eqn{n_i' \times p_i, i = 1, 2, 3}
#' (\eqn{n_i'} is the number of new marginal data points in the \eqn{i}th dimension).
#'
#'
#' @param object An object of class SMME, produced with \code{softmaximin} with
#' \eqn{m_\zeta} fitted models for each value of \code{zeta}.
#' @param x a matrix of size \eqn{n' \times p} with \eqn{n'} is the number of new data points.
#' @param X  a list containing the data matrices each of size \eqn{n'_{i} \times p_i},
#' where \eqn{n'_{i}} is the number of new data points in  the \eqn{i}th dimension.
#' @param ... ignored
#'
#' @return  A list of length \code{length(zeta)}.
#' If new covariate data is supplied as an \eqn{n' \times p} matrix \code{x}
#' each item in the list is an \eqn{n'\times m_\zeta} matrix with the linear
#' predictors computed for each model. If new covariate data is supplied as a
#' list of matrices each of size \eqn{n'_{i} \times p_i},  each item is an array
#' of size \eqn{n'_1 \times \cdots \times n'_d \times m_\zeta},
#' \eqn{d\in \{1,2,3\}}, with the linear predictors computed for each model.
#' @examples
#'
#' ##size of example
#' n1 <- 65; n2 <- 26; n3 <- 13; p1 <- 13; p2 <- 5; p3 <- 4
#'
#' ##marginal design matrices (Kronecker components)
#' X1 <- matrix(rnorm(n1 * p1, 0, 0.5), n1, p1)
#' X2 <- matrix(rnorm(n2 * p2, 0, 0.5), n2, p2)
#' X3 <- matrix(rnorm(n3 * p3, 0, 0.5), n3, p3)
#' X <- list(X1, X2, X3)
#'
#' component <- rbinom(p1 * p2 * p3, 1, 0.1)
#' Beta1 <- array(rnorm(p1 * p2 * p3, 0, 0.1) + component, c(p1 , p2, p3))
#' Beta2 <- array(rnorm(p1 * p2 * p3, 0, 0.1) + component, c(p1 , p2, p3))
#' mu1 <- RH(X3, RH(X2, RH(X1, Beta1)))
#' mu2 <- RH(X3, RH(X2, RH(X1, Beta2)))
#' Y1 <- array(rnorm(n1 * n2 * n3, mu1), dim = c(n1, n2, n3))
#' Y2 <- array(rnorm(n1 * n2 * n3, mu2), dim = c(n1, n2, n3))
#'
#' Y <- array(NA, c(dim(Y1), 2))
#' Y[,,, 1] <- Y1; Y[,,, 2] <- Y2;
#'
#' fit <- softmaximin(X, Y, zeta = c(1, 10), penalty = "lasso", alg = "npg")
#'
#' ##new data in matrix form
#' x <- matrix(rnorm(2 * p1 * p2 * p3), nrow = 2)
#' yhat <- predict(fit, x = x)
#'
#' ##new data in tensor component form
#' X1 <- matrix(rnorm(2 * p1), nrow = 2)
#' X2 <- matrix(rnorm(3 * p2), nrow = 3)
#' X3 <- matrix(rnorm(4 * p3), nrow = 4)
#' Yhat <- predict(fit, X = list(X1, X2, X3))
#'
#' @author Adam Lund
#' @method predict SMME
#' @export
predict.SMME <- function(object, x = NULL, X = NULL, ...){

nzeta <- length(object$zeta)
p <- object$dimcoef
out <- vector("list", nzeta)
names(out) <- object$zeta

if(is.null(x) & is.null(X)){

stop(paste("no new data provided"))

}else if(is.null(x) == FALSE & is.null(X)){

x <- as.matrix(x)
coldim <- dim(x)[2]
nofcoef <- prod(p)

if(coldim != nofcoef){

stop(
paste("column dimension of the new data x (", coldim ,") is not equal to the number of coefficients p (", nofcoef ,")", sep = "")
)

}

for(z in 1:length(object$zeta)){
nlambda <- length(object$lambda[[z]])
res <- matrix(NA ,dim(x)[1], nlambda)
for(i in 1:nlambda){

beta <- object$coef[[z]][ , i]
res[, i] <- x %*% beta

}

out[[z]] <- res

}

}else if(is.null(x)  & is.null(X) == FALSE){

if(!is.list(X)){stop(paste("X must be a list of length 1, 2 or 3!"))}

dimglam <- length(X)

if (dimglam > 3){

stop(paste("the dimension of the model must be 1, 2 or 3!"))

}else if(dimglam == 1){

X[[2]] <- matrix(1, 1, 1)
X[[3]] <- matrix(1, 1, 1)

}else if(dimglam == 2){X[[3]] <- matrix(1, 1, 1)}

X1 <- X[[1]]
X2 <- X[[2]]
X3 <- X[[3]]

dimX <- rbind(dim(X1), dim(X2), dim(X3))

n1 <- dimX[1, 1]
n2 <- dimX[2, 1]
n3 <- dimX[3, 1]
p1 <- dimX[1, 2]
p2 <- dimX[2, 2]
p3 <- dimX[3, 2]
n <- prod(dimX[,1])
p <- prod(dimX[,2])

coldim <- dim(X1)[2] * dim(X2)[2] * dim(X3)[2]

if(coldim != p){

stop(
paste("column dimension of the kronecker product of the new data X (", coldim ,") is not equal to the number of coefficients p (", p ,")", sep = "")
)

}

for(z in 1:length(object$zeta)){

nlambda <- length(object$lambda[[z]])
res <- array(NA , c(n1, n2, n3, nlambda))

for(i in 1:nlambda){

beta <- array(object$coef[[z]][ , i], dim = c(p1, p2, p3))
res[,,, i] <- RH(X3, RH(X2, RH(X1, beta)))

}

out[[z]] <- drop(res)

}

}else{stop(paste("dimension of new data inconsistent with existing data"))}

return(out)

}
