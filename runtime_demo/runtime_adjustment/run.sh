
#!/bin/bash
echo `pwd`

level=4 ##initial level 
constraint=60 ##latency constraint
accuracy=76 ##Accuracy constraint
sum=0
level_list=($level -10)
while [ $level -ge 0 -a $level -le 5 ];
do
    echo $level
    res=`python3 -c 'import test2;print(test2.picknet('${level}','${constraint}','${accuracy}'))'`
    label=${res:0:1}
    p=${res:1}
    echo $label
    #echo $p

    if [ $label -eq 0 ]
    then
        let sum=level_list[1]+1
        if [ $level -eq ${sum} ]
        then
            let level-=1
            echo "The level set as: ${level}"   
            let level_list[0]=$level
            let level_list[1]=$level    
            #let constraint=1000
        else
            let level-=1
        fi
    elif [ $label -eq 1 ]
    then
        let level_list[0]=${level_list[1]}
        let level_list[1]=$level
        let level+=1
    else
        break
    fi
done
    
