import matplotlib.pyplot as plt
import numpy as np

#original size 58

x =[0.0,
-1.860973797281988140e-01,
-1.587807720224054808e-01,
-1.345379660252315757e-01,
-1.138132079901674060e-01,
-9.385550943029191684e-02,
-7.786010508601648450e-02,
-6.266211847487772324e-02,
-4.846849359464676377e-02,
-3.658161094784661421e-02,
-2.528520203641192871e-02,
-1.375149086925809422e-02,
-4.317577937763859147e-03,
5.353637249449517965e-03,
1.333894219654561952e-02,
2.176849339713202852e-02,
2.957721260440495173e-02,
3.717898781390666096e-02,
4.436687502786612614e-02,
5.134781824405439121e-02,
5.791487346470036018e-02,
6.448192868534628752e-02,
7.084203990822107722e-02,
7.699520713332470845e-02,
8.294143036065716734e-02,
8.888765358798962624e-02,
9.462693281755094055e-02,
1.003662120471122271e-01,
1.061054912766735414e-01,
1.116378265084636834e-01,
1.171701617402538115e-01,
1.227024969720439534e-01,
1.292999104506378782e-01,
1.348322456824280480e-01,
1.401576369164470315e-01,
1.456899721482371457e-01,
1.510153633822561292e-01,
1.565476986140462712e-01,
1.618730898480652547e-01,
1.671984810820842382e-01,
1.725238723161032217e-01,
1.778492635501221775e-01,
1.823165205351085227e-01,
1.876419117691275062e-01,
1.929673030031464898e-01,
1.982926942371654733e-01,
2.036180854711844568e-01,
2.089434767052034125e-01,
2.142688679392223960e-01,
2.195942591732413796e-01,
2.238545721604565664e-01,
2.291799633944755499e-01,
2.345053546284945334e-01,
2.398307458625135169e-01,
2.451561370965324727e-01,
2.494164500837476595e-01,
2.547418413177666152e-01,
2.600672325517855987e-01,
5.600672325517855987e-01
]



y = [0.0,
-5.524228089748531856e-01,
-3.832938974504886409e-01,
-2.390255009462969316e-01,
-1.177821093559610621e-01,
-1.562424907682091746e-02,
6.955204338746706627e-02,
1.429099150437741772e-01,
2.038375291899949260e-01,
2.558342143827234683e-01,
2.987282789659202464e-01,
3.349670697480077730e-01,
3.660427135232238172e-01,
3.915150652455022806e-01,
4.130479433651204069e-01,
4.306413478820783625e-01,
4.453472605445477073e-01,
4.573373730085681377e-01,
4.670518303202055299e-01,
4.761544509297373495e-01,
4.819294346387145156e-01,
4.877044183476917372e-01,
4.928675653545632751e-01,
4.963668939111574585e-01,
4.982024040174743984e-01,
5.000379141237912828e-01,
5.012615875280026501e-01,
5.024852609322139063e-01,
5.037089343364251626e-01,
5.032687892903591198e-01,
5.028286442442930770e-01,
5.023884991982270343e-01,
5.019483541521609915e-01,
5.015082091060949487e-01,
5.004562273579232778e-01,
5.000160823118573461e-01,
4.979121188155138933e-01,
4.974719737694479060e-01,
4.964199920212762351e-01,
4.943160285249328934e-01,
4.932640467767612225e-01,
4.911600832804178807e-01,
4.907199382343518379e-01,
4.896679564861801670e-01,
4.875639929898368252e-01,
4.865120112416651543e-01,
4.844080477453218125e-01,
4.833560659971501416e-01,
4.812521025008067999e-01,
4.802001207526351290e-01,
4.780961572562917872e-01,
4.770441755081201163e-01,
4.749402120117767745e-01,
4.738882302636051036e-01,
4.717842667672617618e-01,
4.707322850190900909e-01,
4.686283215227467491e-01,
4.675763397745751337e-01,
5.600672325517855987e-01
]

print(len(x))
print(len(y))
start =  np.array([0.,0.,0.,0.]) #x,y,dx,dy
goal  =  np.array([.5682151,   0.5682151,   0.36304926,  0.36304926]) #x,y,dx,dy
plt.clf()
#plt.subplot(221)
plt.scatter(start[0], start[1], color='r')
plt.scatter(goal[0],  goal[1],  color='g')

plt.plot(x, y, color='b')
plt.xlim([-0.1, 1.])
plt.ylim([-0.1, 1.])
plt.xlabel("X location")
plt.ylabel("Y location")
plt.show()