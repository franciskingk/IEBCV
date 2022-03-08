# IEBCV
computer vision project
Voting Tally Authentication

# Overview
Elections in Kenya usually occur every 5 years. The last election took place in the year 2017. Elections in Kenya take place within the framework of a multi-party democracy and a presidential system. The President, Senate and National Assembly are directly elected by voters, with elections organized by the Independent Electoral and Boundaries Commission (IEBC). The tallying of the votes is done manually by returning officers located in every polling station. 

# Problem statement
In the previous elections, it is common to have the sentiment among voters divided as the losing party often doesn’t accept the election outcomes. As the tallying is undertaken manually, recounts are usually not done unless requested, therefore there is no definitive way to validate all the votes. It is also easier for voters to claim the returning process is biased since the votes are hand counted. 

# Proposed solution
To authenticate the voting tally, we propose to build a computer vision model that will scan the unique identifier, identify if a ballot is correctly filled, identify the candidate that has acquired the vote and record the tally in a csv file.

# Justification
We wish to create a non-biased model that will assist in the validation of vote tallies. By doing this we wish to eliminate doubt in the tallying process and hence reduce tension that might exist between citizens that voted for different candidates. 
The government staff responsible for tallying will also achieve more with less time as manual recounting of votes will no longer be necessary. 
The voting tallies will also be viewable in real time as the model counts the votes for the candidates, hence deterring claims of election fraud.

# Objectives
# Main objective
To use computer vision to tally votes and detect invalid ballot papers.

# Specific objectives
To identify the qr code on each ballot paper to validate it as a real government provided ballot paper and identify which area the vote originated from.
To correctly classify the invalid and valid votes.
To correctly identify the ballot papers as either presidential, gubernatorial, senate, ward ballot,  women representatives or parliamentary.
To tally the votes for the candidates chosen and store the data in a csv file.

# Success Criteria
To consider our project successful we should have been able to create a model that can undertake the following tasks:
Read the serial number on the ballot paper.
Identify which area the ballot paper originates from.
Classify a vote as invalid or valid.
Identify each ballot paper as either presidential, gubernatorial, senate, ward, women rep or parliamentary.
Tally the candidate votes and records the data in a csv file.

# Data wrangling
We were able to obtain sample ballot papers for the different government positions. The ballot papers were representative of the 2017 elections ballot papers.


# Data Understanding
Our project will focus on two polling stations as a case study which will be;
One polling station in Nairobi county, Dagoretti North specifically in Nairobi Primary School.
Another polling station will be in Mombasa county, Changamwe specifically Bomu Primary School.

There were 2,250,853 registered voters in Nairobi county in 2017 while the number of polling stations were 3,378 therefore we settled on having 600 ballot papers for the Nairobi school polling station. In Mombasa county, there were 580,223 registered voters with 934 polling stations therefore, we settled on having 600 ballot papers for the Bomu polling station.
# Data Preparation
Created several functions to:
Resize the image so that all of them have the same pixel count. This will make it easier to calibrate the coordinates of shapes in all ballot paper images.
Create several regions of interest on the image. Here we will have square regions of interest in every box within the marking column .
Processing the image that will be fed into the mark recognition function and making the color binary. This will help us check the pixel count in order to determine the box with the higher pixel count.
Make text thicker (dilation).
Use tesseract to extract the text within the region of interest which in this case is the one that has been marked.

# Modeling and evaluation
Built a model using open cv to:
Check the marks on the ballot papers
Specify the coordinates for our marked bounding boxes
Cropping the image and computing the number of pixels in each
Determine where the ballot is marked by investigating the count for pixels (we set a condition that where the number of pixels exceeds 100, the ballot paper is considered marked).
If more than one region is marked, then the vote is considered to be spoiled.
Obtain the marked checked image, name of the candidate chosen, whether the vote is spoiled or not and the elective position the ballot paper is for.




# Recommendations

The model will work with image capture and will therefore require access to a stable camera with good resolution. This will enable the real time input and tally of the ballot papers.
As a way to avoid counting one vote multiple times, each ballot paper can have a unique identifier that will be in the IEBCV database so that once a ballot paper is scanned, the serial number or unique identifier related to it will be locked hence avoiding another count of the same vote.

# Conclusion 

Having gone through the various stages of the project, the team has been able to meet the objectives against all odds. The project is a success and will be open for more improvements over time.

