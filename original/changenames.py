# Pythono3 code to rename multiple
# files in a directory or folder
 
# importing os module
import os
 
# Function to rename multiple files
def main():

    for i in range(8,30):
        folder = str(i)

        for count, filename in enumerate(os.listdir(folder)):
            # dst = f"Hostel {str(count)}.jpg"
            src =f"{folder}/{filename}"  # foldername/filename, if .py file is outside folder
            dst =f"{folder}/{count+1}.jpg"
             
            # rename() function will
            # rename all the files
            os.rename(src, dst)
 
# Driver Code
if __name__ == '__main__':
     
    # Calling main() function
    main()