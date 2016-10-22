from math import  radians, sqrt, tan
# constants from the Iphone: FOV field of view in degrees, and number of pixels wide & tall in photo
FOVx=63.54
FOVy=44.95886879
PIC_LENGTH=3264.
PIC_HEIGHT=2448.

def dist_to_pitcher(p, c, pitcher_catcher_distance=60.5, h=5.):
    """
    p & c are tuples of 2 values-- the x-y coordinates of the pitcher and catcher respectively (Starting from top left corner)
    pitcher_catcher_distance is the distance (in reality) from the pitcher to the catcher in feet -- 60.5 for regulation baseball field
    h is the height off the ground that the pitcher holds the camera -- for our purposes we can assume 5 ft. on average(output not that sensitive)
    
    y coordinates are not used -- good predictions without
    """
    
    # I took empirical measurements from the right side of the pitcher if he is facing the catcher
    # so, if the picture is taken from the other side, we must switch x coordinates to represent this setup 
    # (this implicitly assumes the coefficients we estimated are true for both sides -- still need to verify)
    if p[0] > c[0]:
        p = (PIC_LENGTH - p[0], p[1])
        c = (PIC_LENGTH - c[0], c[1])
    
    # number of x pixels and y pixels between the pitcher and catcher
    x_pixels = abs(p[0] - c[0])
    
    # convert pixels to angle (in radians)
    x_proportion = x_pixels / PIC_LENGTH
    x_angle = radians(x_proportion*FOVx)
    
    # calculate distance with trig
    distance_ft = pitcher_catcher_distance / tan(x_angle)
    
    # this corrects for h -- right triangle: hypotenuse is distance_ft & h is "short side", we want "long side" that is on the ground
    d = sqrt(distance_ft**2 - h**2)
    
    ## d got us close to the correct distance when measured empirically, but not good enough, so we made a linear model of the
    ## difference between d and the real distance that was measured with a measuring tape
    
    ## that linear model takes as inputs:
    ##   d (calculated above)
    ##   x_pixels*p[0] the interaction
    ##   x_pixels
    ##   p[0] -- the x coordinate of the pitcher (this assumes that we are on the "right" hand side of the pitcher if he is looking at the catcher)
    ##   p[0]**2 the x coordinate of the pitcher squared
    ##   p[0]**4 the x coordinate of the pitcher to the 4th power
    ##   an intercept term
    
    # hard coded coefficients as estimated from a sample of photos
    coefficients = [1.2724175616337765, 
                    1.5722041791536374e-05,
                    0.0091480432806466325,
                    -0.04726140971001247,
                    1.2019398669013432e-05,
                    9.1210451413941492e-13,
                    -33.503517724311536
                   ]
    
    independent_vars = [d, x_pixels*p[0], x_pixels, p[0], p[0]**2, p[0]**4, 1]
    
    # dot product
    predicted_distance = sum(var*coef for var, coef in zip(independent_vars, coefficients))
    
    return predicted_distance