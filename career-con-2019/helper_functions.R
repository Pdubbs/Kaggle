quaternion_convert <- function(x, y, z, w){
  t0 <- 2*(w*x + y*z)
  t1 <- 1.0-2.0*(x^2 + y^2)
  ex <- atan2(t0, t1)
  
  t2 <- 2.0*(w*y - z*x)
  t2 <- ifelse(t2>1, 1, t2)
  t2 <- ifelse(t2<-1, -1,  t2)
  ey <- asin(t2)
  
  t3 <- 2*(w*z + x*y)
  t4 <- 1-2*(y^2 + z^2)
  ez <- atan2(t3, t4)
  return(c(ex,ey,ez))
}

quaternion_mat <- function(mat, var){
  vars <- paste0(var,'_',c('X','Y','Z','W'))
  euler <- data.frame(t(apply(mat[,vars],1
                            ,function(x) quaternion_convert(x[1],x[2],x[3],x[4]))))
  colnames(euler) <- paste0(var,'_euler_',c('X','Y','Z'))
  return(euler)
}

abs_range <- function(x){
  return(max(x)-min(x))
}

make_trainvars <- function(mat) {
  euler_orient <- quaternion_mat(mat,'orientation')
  mat <- cbind(mat,euler_orient)
  mat$abs_acceleration <- sqrt(mat$linear_acceleration_X^2 +
                                     mat$linear_acceleration_Y^2 +
                                     mat$linear_acceleration_Z^2)
  mat$abs_velocity <- sqrt(mat$angular_velocity_X^2 +
                           mat$angular_velocity_Y^2 +
                           mat$angular_velocity_Z^2)
  mat$acc_over_vel <- mat$abs_acceleration/mat$abs_velocity
  mat$acc_dot_orient <- mat$orientation_euler_X*mat$linear_acceleration_X +
                        mat$orientation_euler_Y*mat$linear_acceleration_Y +
                        mat$orientation_euler_Z*mat$linear_acceleration_Z
  mat$acc_orient_ang <- acos(mat$acc_dot_orient/(sqrt(mat$orientation_euler_X^2+
                                                      mat$orientation_euler_Y^2+
                                                      mat$orientation_euler_Z^2)*
                                                 sqrt(mat$linear_acceleration_X^2+
                                                      mat$linear_acceleration_Y^2+
                                                      mat$linear_acceleration_Z^2)
                                                 )
                             )
  return(mat)
}
