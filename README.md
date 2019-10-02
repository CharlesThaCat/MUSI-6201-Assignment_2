# MUSI-6201-Assignment_2

## Answer for Question 2 in Section C
RMS-mean, RMS-std pair works the best for separating music and speech. This is because in music samples, the RMS seldom goes down to zero and does not diverge largely from the mean RMS, while in speech samples, there are relatively many frames containing zero or close to zero RMS, and the changes are rapid. <br>
SF-mean, ZCR-mean pair can hardly be used in effective music-speech separation because the red and blue dots largely blend together.<br>
ZCR-std, SCR-std and SC-std, SF-std pairs can provide a certain amount of separation, but not as good as RMS-mean, RMS-std pair, since the boundaries between blue and red dots are not as clear as that in RMS-mean, RMS-std pair.