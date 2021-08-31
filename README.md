# FilterApplication Readme File
## The project is the second part of the TweetFiltered project. The first half accesses twitter posts and stores them to an API deployed to GCP with AppEngine - more information you can find [HERE](https://github.com/vilkata41/tweetfilteredAPI).

The project consists only of a single python script that carries out the main functionality of the face filter application. The rest of the files in the project are images, folders that contain downloaded images, and ones that contain the final products. <br><br>

We start off the program with instantiating the API already creaded in the TweetFilteredAPI program and getting the response in a json format.<br>
After that, we download the pictures into the ```imgs``` folder and the videos into the ```vids``` folder *(if there is any media, of course - if there is no media, we skip the post)*.<br>
I've included two pictures that one could use for testing purposes if they wouldn't want to use the API but rather just use their own pictures. I've also included a filter - an Iron Man helmet.<br>
We append every image to a list called ```IMAGE_FILES``` so that we could later iterate through it and apply the filters to all pictures we have in there.<br> <br>

Before applying the filter, however, I save two different types of picture - one with all  the facial landmarks, and one with a rectangle marking the dimensions of the face.<br>
The final part is rather quick - we resize the filter according to the facial dimensions and we add it as an overlay.<br><br>

The end products *(annotated images, face marked images, face filtered images)* are stored in the ```tmp``` folder in the project.<br><br>

```There are some comments in the project that I've left - they could be uncommented for testing purposes.```
