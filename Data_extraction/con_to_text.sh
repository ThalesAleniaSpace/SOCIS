
for i in * ; do
    temp = $i
    echo $temp
    DIR = '/temp'
    echo $DIR
    # if [ "$(ls -A $DIR)" ]; then
        #FOR THE CONVERSION OF sfg FILES
        # if ["$(find . -type f -name '$i/*.sfg')"]; then

            for file in $i/*.sfg; do
                mv "$file" "${file%.sfg}.txt"
            done
        # fi
        
    	#FOR THE CONVERSION OF SFI FILES
        
        # if ["$(find . -type f -name '$i/*.SFI')"];then

            for file in $i/*.SFI; do
                mv "$file" "${file%.SFI}.txt"
            done
        # fi

    # fi
done
 
