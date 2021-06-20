#     Description of this R script:
#     R interface/wrapper for the Rcpp function pga in the SMME package.
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
#' @name SMME
#' @aliases pga
#' @title Soft Maximin Estimation for Large Scale Data
#'
#' @description  Efficient procedure for solving the Lasso or SCAD penalized soft
#' maximin problem for large scale data. This software implements two proximal
#' gradient based algorithms (NPG and FISTA) to solve different forms of the soft
#' maximin problem from, see \cite{Lund et al., 2020}. For general group specific
#' design the soft maximin problem is solved using the NPG algorithm.
#' For d = 1,2 or 3 and fixed identical d-array-tensor design in all groups the
#' estimation procedure uses either the FISTA algorithm or the NPG algorithm.
#' Note for d = 2 or 3 the procedure avoids the tensor design matrix.
#' @usage  softmaximin(x,
#'             y,
#'             zeta,
#'             penalty = c("lasso", "scad"),
#'             alg = c("npg", "fista"),
#'             nlambda = 30,
#'             lambda.min.ratio = 1e-04,
#'             lambda = NULL,
#'             penalty.factor = NULL,
#'             reltol = 1e-05,
#'             maxiter = 15000,
#'             steps = 1,
#'             btmax = 100,
#'             c = 0.0001,
#'             tau = 2,
#'             M = 4,
#'             nu = 1,
#'             Lmin = 0,
#'             log = TRUE,
#'             nthreads = 4)
#'
#' @param x list containing the G group specific design matrices of sizes
#' \eqn{n_i \times p_i}. Alternatively for a model with identical tensor design
#' across G groups, \code{x} is a list containing the \eqn{d}
#' (\eqn{d \in \{ 1, 2, 3\}}) tensor components.
#' @param y list containing the G group specific response vectors of sizes
#' \eqn{n_i \times 1}. Alternatively for a model with identical tensor design
#' across G groups, \code{y}is an array of size
#' \eqn{n_1 \times\cdots\times n_d \times G}  (\eqn{d \in \{ 1, 2, 3\}}) containing the response values.
#' @param zeta strictly positive float controlling  the softmaximin approximation
#' accuracy.
#' @param penalty string specifying the penalty type. Possible values are
#' \code{"lasso", "scad"}.
#' @param alg string specifying the optimization algorithm. Possible values are
#' \code{"npg", "fista"}.
#' @param nlambda positive integer giving the number of \code{lambda} values.
#' Used when lambda is not specified.
#' @param lambda.min.ratio strictly positive float giving the smallest value for
#' \code{lambda}, as a fraction of \eqn{\lambda_{max}}; the (data dependent)
#' smallest value for which all coefficients are zero. Used when lambda is not
#' specified.
#' @param lambda sequence of strictly positive floats used  as penalty parameters.
#' @param penalty.factor array of size \eqn{p_1 \times \cdots \times p_d} of
#' positive floats. Is multiplied with each element in \code{lambda} to allow
#' differential penalization on the coefficients.
#' @param reltol strictly positive float giving the convergence tolerance for
#' the inner loop.
#' @param maxiter positive integer giving the maximum number of  iterations
#' allowed for each \code{lambda} value, when  summing over all outer iterations
#' for said \code{lambda}.
#' @param steps strictly positive integer giving the number of steps used in the
#' multi-step adaptive lasso algorithm for non-convex penalties. Automatically
#' set to 1 when \code{penalty = "lasso"}.
#' @param btmax strictly positive integer giving the maximum number of backtracking
#' steps allowed in each iteration. Default is \code{btmax = 100}.
#' @param c strictly positive float used in the NPG algorithm. Default is
#' \code{c = 0.0001}.
#' @param tau strictly positive float used to control the stepsize for NPG.
#' Default is \code{tau = 2}.
#' @param M positive integer giving the look back for the NPG. Default is \code{M = 4}.
#' @param nu strictly positive float used to control the stepsize. A  value less
#' that 1 will decrease the stepsize and a value larger than one will increase it.
#' Default is \code{nu = 1}.
#' @param Lmin non-negative float used by the NPG algorithm to control the
#' stepsize. For the default  \code{Lmin = 0} the maximum step size is the same
#' as for the FISTA algorithm.
#' @param log logical variable indicating whether to use log-loss.  TRUE is
#' default and yields the loss below.
#'
#' @details Consider modeling heterogeneous data \eqn{y_1,\ldots, y_n} by dividing
#' it into \eqn{G} groups \eqn{\mathbf{y}_g = (y_1, \ldots, y_{n_g})},
#' \eqn{g \in \{ 1,\ldots, G\}} and then using a linear model
#' \deqn{
#' \mathbf{y}_g = \mathbf{X}_gb_g + \epsilon_g, \quad g \in \{1,\ldots, G\},
#' }
#' to model the group response. Then \eqn{b_g} is a group specific \eqn{p\times 1}
#' coefficient, \eqn{\mathbf{X}_g} an \eqn{n_g\times p} group design matrix and
#' \eqn{\epsilon_g} an \eqn{n_g\times 1} error term. The objective is to estimate
#' a common coefficient \eqn{\beta} such that \eqn{\mathbf{X}_g\beta} is a robust
#' and good approximation to \eqn{\mathbf{X}_gb_g} across groups.
#'
#' Following \cite{Lund et al., 2020}, this objective may be accomplished by
#' solving the soft maximin estimation problem
#' \deqn{
#' \min_{\beta}\frac{1}{\zeta}\log\bigg(\sum_{g = 1}^G \exp(-\zeta \hat V_g(\beta))\bigg)
#'  + \lambda  \Vert\beta\Vert_1, \quad \zeta > 0,\lambda \geq 0.
#' }
#' Here \eqn{\zeta} essentially controls the amount of pooling across groups
#' (\eqn{\zeta \sim 0} ignores grouping and pools observations) and
#' \deqn{
#' \hat V_g(\beta):=\frac{1}{n_g}(2\beta^\top \mathbf{X}_g^\top
#' \mathbf{y}_g-\beta^\top \mathbf{X}_g^\top \mathbf{X}_g\beta),
#' }
#' is the empirical explained variance from \cite{Meinshausen and B{u}hlmann, 2015}.
#' See \cite{Lund et al., 2020} for more details and references.
#'
#' The function \code{softmaximin} solves the soft maximin estimation problem in
#' large scale settings for a sequence of penalty parameters
#' \eqn{\lambda_{max}>\ldots >\lambda_{min}>0}. The implementation also solves the
#' problem above with the penalty given by the SCAD penalty, using the multiple
#' step adaptive lasso procedure to loop over the inner proximal algorithm.
#'
#' Two optimization algorithms  are implemented in the SMME packages;
#' a non-monotone proximal gradient (NPG) algorithm and a fast iterative soft
#' thresholding algorithm (FISTA). The implementation is particularly efficient
#' for applications where the design is identical across groups i.e. \eqn{\mathbf{X}_g = \mathbf{X}}
#' \eqn{\forall g \in \{1, \ldots, G\}} and where \eqn{\mathbf{X}} has tensor
#' structure i.e.
#' \deqn{
#' \mathbf{X} = \bigotimes_{i=1}^d \mathbf{M}_i.
#' }
#' for marginal \eqn{n_i\times p_i} design matrices \eqn{\mathbf{M}_1,\ldots, \mathbf{M}_d}.
#'
#' For \eqn{d \in \{ 1, 2, 3\}}, provided only with the marginal matrices and the
#' group response vectors, \code{softmaximin} solves the soft maximin problem with
#' minimal memory footprint using tensor optimized arithmetic, see  \code{\link{RH}}.
#'
#' @return An object with S3 Class "SMME".
#' \item{spec}{A string indicating the array dimension (1, 2 or 3) and the penalty.}
#' \item{coef}{A \eqn{p_1\cdots p_d \times} \code{nlambda} matrix containing the
#' estimates of the model coefficients (\code{beta}) for each \code{lambda}-value.}
#' \item{lambda}{A vector containing the sequence of penalty values used in the
#' estimation procedure.}
#' \item{Obj}{A matrix containing the objective values for each iteration and
#' each model.}
#' \item{df}{The number of nonzero coefficients for each value of \code{lambda}.}
#' \item{dimcoef}{A vector giving the dimension of the model coefficient array
#' \eqn{\beta}.}
#' \item{dimobs}{A vector giving the dimension of the observation (response)
#' array \code{Y}.}
#' \item{Iter}{A list with 4 items:
#' \code{bt_iter}  is total number of backtracking steps performed,
#' \code{bt_enter} is the number of times the backtracking is initiated,
#' and \code{iter_mat} is a vector containing the  number of  iterations for each
#' \code{lambda} value and  \code{iter} is total number of iterations.}
#'
#' @author  Adam Lund
#'
#' Maintainer: Adam Lund, \email{adam.lund@@math.ku.dk}
#'
#' @references
#' Lund, A., S. W. Mogensen and N. R. Hansen (2020). Soft Maximin Estimation for
#' Heterogeneous Array Data. \emph{Preprint}.
#'
#' Meinshausen, N and P. B{u}hlmann (2015). Maximin effects in inhomogeneous
#' large-scale data. \emph{The Annals of Statistics}. 43, 4, 1801-1830.
#' url = {https://doi.org/10.1214/15-AOS1325}.
#'
#' @keywords package
#'
#' @examples
#' #Non-array data
#'
#' ##size of example
#' set.seed(42)
#' G <- 10; n <- sample(500:1500, G); p <- 60
#' x <- y <- list()
#'
#' ##group design matrices
#' for(g in 1:G){x[[g]] <- matrix(rnorm(n[g] * p), n[g], p)}
#'
#' ##common features and effects
#' common_features <- rbinom(p, 1, 0.1)
#' common_effects <- rnorm(p) * common_features
#'
#' ##group response
#' for(g in 1:G){
#' bg <- rnorm(p, 0, 0.5) * (1 - common_features) + common_effects
#' mu <- x[[g]] %*% bg
#' y[[g]] <- rnorm(n[g]) + mu
#' }
#'
#' ##fit model for specific zeta
#' system.time(fit <- softmaximin(x, y, zeta = c(0.1, 1, 10 , 100), penalty = "lasso", alg = "npg"))
#' Betafit <- fit$coef
#'
#' ##estimated common effects for specific lambda
#' modelno <- 6
#' m <- min(Betafit[ , modelno], common_effects)
#' M <- max(Betafit[ , modelno], common_effects)
#' plot(common_effects, type = "p", ylim = c(m, M), col = "red")
#' lines(Betafit[ , modelno], type = "h")
#'
#' #Array data
#'
#' ##size of example
#' set.seed(42)
#' G <- 5; n <- c(65, 26, 13); p <- c(13, 5, 4)
#'
#' ##marginal design matrices (Kronecker components)
#' x <- list()
#' for(i in 1:length(n)){x[[i]] <- matrix(rnorm(n[i] * p[i]), n[i], p[i])}
#'
#' ##common features and effects
#' common_features <- rbinom(prod(p), 1, 0.1)
#' common_effects <- rnorm(prod(p), 0, 0.1) * common_features
#'
#' ##group response
#'  y <- array(NA, c(n, G))
#' for(g in 1:G){
#' bg <- rnorm(prod(p), 0, 0.1) * (1 - common_features) + common_effects
#' Bg <- array(bg, p)
#' mu <- RH(x[[3]], RH(x[[2]], RH(x[[1]], Bg)))
#' y[,,, g] <- array(rnorm(prod(n)), dim = n) + mu
#' }
#'
#' ##fit model for specific zeta
#' system.time(fit <- softmaximin(x, y, zeta = c(0.1, 1, 10, 100), penalty = "lasso", alg = "npg"))
#' Betafit <- fit$coef
#'
#' ##estimated common effects for specific lambda
#' modelno <- 10
#' m <- min(Betafit[, modelno], common_effects)
#' M <- max(Betafit[, modelno], common_effects)
#' plot(common_effects, type = "h", ylim = c(m, M), col = "red")
#' lines(Betafit[, modelno], type = "h")
#'
#' @export
#' @useDynLib SMME, .registration = TRUE
#' @importFrom Rcpp evalCpp
softmaximin <- function(
               x, #G list or d list
               y, #G list or d array
               zeta,
               penalty = c("lasso", "scad"),
               alg = c("npg", "fista"),
               nlambda = 30,
               lambda.min.ratio = 1e-04,
               lambda = NULL,
               penalty.factor = NULL,
               reltol = 1e-05,
               maxiter = 15000,
               steps = 1,
               btmax = 100,
               c = 0.0001,
               tau = 2,
               M = 4,
               nu = 1,
               Lmin = 0,
               log = TRUE,
               nthreads = 4){

if(sum(alg == c("npg", "fista")) != 1){stop(paste("algorithm must be correctly specified"))}

if(alg == "npg"){alg <- 1}else{alg <- 0}

if(log == TRUE){ll <- 1}else{ll <- 0}

if(c <= 0){stop(paste("c must be strictly positive"))}

if(Lmin < 0){stop(paste("Lmin must be positive"))}

if(mean(zeta <= 0) > 0){stop(paste("all zetas must be strictly positive"))}

if(sum(penalty == c("lasso", "scad")) != 1){stop(paste("penalty must be correctly specified"))}

if(!is.null(penalty.factor)){
if(min(penalty.factor) < 0){stop(paste("penalty.factor must be positive"))}
}

if(class(y) == "list"){

  Z <- lapply(y, as.matrix)
  rm(y)
  array = 0
  dimglam <- NULL
  G = length(Z)
  n <- sum(sapply(Z, length))
  p <- dim(x[[1]])[2]
  alg = 1 #npg
  if(is.null(penalty.factor)){penalty.factor <- as.matrix(rep(1, p))
  }else{penalty.factor <- as.matrix(penalty.factor)}
  #check to make sure y is compaitble with x in every gropu...todo

}else if(class(y) == "array"){

array = 1
dimglam <- length(x)

if(dimglam != length(dim(y)) - 1){stop(paste("x and y not compatible"))}

if (dimglam > 3){

stop(paste("the dimension of the model must be 1, 2 or 3!"))

}else if (dimglam == 1){

x[[2]] <- matrix(1, 1, 1)
x[[3]] <- matrix(1, 1, 1)

}else if (dimglam == 2){

x[[3]] <- matrix(1, 1, 1)

}

dimx <- rbind(dim(x[[1]]), dim(x[[2]]), dim(x[[3]]))

n1 <- dimx[1, 1]
n2 <- dimx[2, 1]
n3 <- dimx[3, 1]
p1 <- dimx[1, 2]
p2 <- dimx[2, 2]
p3 <- dimx[3, 2]
n <- prod(dimx[,1])
p <- prod(dimx[,2])
G <- dim(y)[length(dim(y))]

Z <- list()

for(i in 1:G){

if(dimglam == 1){

tmp <- matrix(y[,  i], n1, n2 * n3)

}else if(dimglam == 2){

tmp <- matrix(y[, ,  i], n1, n2 * n3)

}else if(dimglam == 3){

tmp <- matrix(y[, , , i], n1, n2 * n3)

}

Z[[i]] <- tmp

}

if(is.null(penalty.factor)){penalty.factor <- matrix(1, p1, p2 * p3)}

}

if(length(penalty.factor) != p){

stop(
paste("number of elements in penalty.factor (", length(penalty.factor),") is not equal to the number of coefficients (", p,")", sep = "")
)

}

if(penalty == "lasso"){steps <- 1}

if(is.null(lambda)){

makelamb <- 1
lambda <- rep(NA, nlambda)

}else{

makelamb <- 0
nlambda <- length(lambda)

}

res <- pga(x,
           Z,
           penalty,
           zeta,
           c,
           lambda, nlambda, makelamb, lambda.min.ratio,
           penalty.factor,
           reltol,
           maxiter,
           steps,
           btmax,
           M,
           tau,
           nu,
           alg,
           array,
           ll,
           Lmin,
           nthreads)

#todo: checks on all stops from each zeta
# if(res$Stops[2] == 1){
#
# warning(paste("program exit due to maximum number of inner iterations (",maxiter,") reached for model no ",res$endmodelno + 1,""))
#
# }
#
# if(res$Stop[3] == 1){
#
# warning(paste("program exit due to maximum number of backtraking steps reached for model no ",res$endmodelno + 1,""))
#
# }

# Iter <- res$Iter
#
# maxiterpossible <- sum(Iter > 0)
# maxiterreached <- sum(Iter >= (maxiter - 1))
#
# if(maxiterreached > 0){
#
# warning(
# paste("maximum number of inner iterations (",maxiter,") reached ",maxiterreached," time(s) out of ",maxiterpossible," possible")
# )
#
#}

out <- list()

class(out) <- "SMME"
out$array = array
if(array){
out$spec <- paste(dimglam,"-dimensional", penalty," penalized array model with", G , "groups")
}else{
out$spec <- paste( penalty," penalized linear model with", G , "groups")
}

##should be looped over and sent to list
endmodelno <- drop(res$endmodelno) #converged models since c++ is zero indexed
if(length(zeta) > 1){
coef <- lambda <- df <- list()
for(z in 1:length(zeta)){

coef[[z]] <- res$Beta[ , 1:endmodelno[z], z]
lambda[[z]] <- res$lambda[1:endmodelno[z], z]
df[[z]] <- res$df[1:endmodelno[z], z]

}

}else{

  coef <- res$Beta[ , 1:endmodelno, 1]
  lambda <- res$lambda[1:endmodelno, 1]
  df <- res$df[1:endmodelno, 1]

}


out$coef <- coef
out$lambda <- lambda
out$df <- df

if(array == 1){

out$dimcoef <- c(p1, p2, p3)[1:dimglam]
out$dimobs <- c(n1, n2, n3)[1:dimglam]

}else{

  out$dimcoef <- p
  out$dimobs <- n

}


#out$Obj <- drop(res$Obj)
#out$endno <- endmodelno

#out$L <- drop(res$L)remonve
#out$L1 <- res$L1 #remove todo
#out$sumsqdiff <- res$Sumsqdiff #todo remove
#out$Delta <- res$Delta #todo remove
#out$deltamax <- res$deltamax

#out$BT <- drop(res$BT)

Iter <- list()
Iter$bt_enter <- drop(res$btenter)
Iter$bt_iter <- drop(res$btiter)
#Iter$iter_mat <- drop(res$Iter[1:endmodelno, ])
Iter$iter <- sum(Iter$iter_mat, na.rm = TRUE)

out$Iter <- drop(Iter)

#out$F <- res$F

return(out)

}

