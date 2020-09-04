import os
from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import numpy as np
from scipy import ndimage
from scipy.ndimage.filters import convolve
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
import math
import operator


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg','bmp'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        flash('Image successfully uploaded and displayed')
        flash('Step1: Applying Canny Edge for edge detection.')
        flash('Step2: Applying Hough Transform')
        flash('Step3: Finding Corners')
        flash('Step4: Finding Quardilateral')
        flash('Final Cropped Image\n')
        return render_template('upload.html', filename=filename)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, bmp')
        return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):

    class HoughEdge():
        def __init__(self,angle,rho,val):
            self.angle=int(angle)
            self.rho=int(rho)
            self.val=int(val)

    class Line():
        def __init__(self,m,b,dist=0,x0=0,y0=0,
                     x1=0,y1=0,end_point=0):
            self.m=m
            self.b=b
            self.dist=int(dist)
            self.x0=int(x0)
            self.y0=int(y0)
            self.x1=int(x1)
            self.y1=int(y1)
            self.end_point=int(end_point)

    class Point():
        def __init__(self,x,y):
            self.x=int(x)
            self.y=int(y)



    def rgb2gray(rgb):
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.3 * r + 0.59 * g + 0.11 * b
        return gray


    # # Step 1.1 Noise Reduction

    # In[4]:


    #Noise reduction
    #Apply Gaussian Filter to smooth the image
    #sigma as 1.4 is just an assumption value
    def gaussian_filter(n,sig=1.4):
        n=int(n)//2
        x,y=np.mgrid[-n:n+1,-n:n+1]
        gauss=(1/(2.0* np.pi * sig**2))*np.exp(-((x**2 + y**2) / (2.0*sig**2)))
        return gauss


    # # Step1.2 Gradient calculation

    # In[5]:



    def filter_sobel(smooth_img):
        #Kx,Ky sobel filter 
        #to get both intensity and edge direction matrices
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
        Ix = convolve(smooth_img, Kx)
        Iy = convolve(smooth_img, Ky)
        G = np.hypot(Ix, Iy)
        G = G / G.max() * 255
        theta = np.arctan2(Iy, Ix)
        return (G, theta)


    # # Step1.3 Non-maximum suppression

    # In[6]:


    def non_max_suppression(edge_img, D):
        #step1
        M, N = edge_img.shape
        #step2
        non_max = np.zeros((M,N), dtype=np.int32)
        #step3- converting negative angle to positive by addgin 180
        angle = D * 180. / np.pi
        angle[angle < 0] += 180


        for i in range(1,M-1):
            for j in range(1,N-1):
                try:
                    q = 255
                    r = 255

                   #angle 0
                    if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                        q = edge_img[i, j+1]
                        r = edge_img[i, j-1]
                    #angle 45
                    elif (22.5 <= angle[i,j] < 67.5):
                        q = edge_img[i+1, j-1]
                        r = edge_img[i-1, j+1]
                    #angle 90
                    elif (67.5 <= angle[i,j] < 112.5):
                        q = edge_img[i+1, j]
                        r = edge_img[i-1, j]
                    #angle 135
                    elif (112.5 <= angle[i,j] < 157.5):
                        q = edge_img[i-1, j-1]
                        r = edge_img[i+1, j+1]

                    if (edge_img[i,j] >= q) and (edge_img[i,j] >= r):
                        non_max[i,j] = edge_img[i,j]
                    else:
                        non_max[i,j] = 0


                except IndexError as e:
                    pass

        return non_max


    # # Step1.4 Double Threshold

    # In[7]:


    #This helps us finding the strong,weak,and irrelevant pixels
    #The pixels other than strong and weak are irrelelavant pixels
    def double_threshold(non_max_image):
        #high threshold
        h_thr=non_max_image.max()*0.17
        #low threshold
        l_thr=h_thr*0.09

        M, N = non_max_image.shape
        result= np.zeros((M,N), dtype=np.int32)

        weak_pixel = np.int32(100)
        strong_pixel = np.int32(255)
        #getting the indices of strong pixels
        strong_i, strong_j = np.where(non_max_image >= h_thr)

        #getting the indices of weak pixels
        weak_i, weak_j = np.where((non_max_image <= h_thr) & (non_max_image >= l_thr))

        result[strong_i, strong_j] = strong_pixel
        result[weak_i, weak_j] = weak_pixel

        return result


    # # Step1.5 Edge Tracking by Hysteresis

    # In[8]:


    #highlight only strong edges
    #checks if any strong pixel is sourrounded by 
    #the weak pixel and updates the pixel value
    #based on the presence of strong pixel
    def edge_hysteresis(image):

        M, N = image.shape
        weak_pixel = 100
        strong_pixel = 255

        for i in range(1, M-1):
            for j in range(1, N-1):
                if (image[i,j] == weak_pixel):
                    flag=False
                    try:
                        if(image[i+1, j-1] == strong_pixel):
                            flag=True
                        if(image[i+1, j] == strong_pixel):
                            flag=True
                        if(image[i+1, j+1] == strong_pixel):
                            flag=True
                        if(image[i, j-1] == strong_pixel):
                            flag=True
                        if(image[i, j+1] == strong_pixel):
                            flag=True
                        if(image[i-1, j-1] == strong_pixel):
                            flag=True
                        if(image[i-1, j] == strong_pixel):
                            flag=True
                        if(image[i-1, j+1] == strong_pixel):
                            flag=True
                        if(flag):
                            image[i, j]=strong_pixel
                        else:
                            image[i, j] = 0
                    except IndexError as e:
                        pass

        return image


    # # Edge Detection

    # In[9]:


    #input file
    
    file_url=os.path.join(app.config['UPLOAD_FOLDER'],filename)
    #url_for('static', filename='uploads/' + filename)
    img_rgb = mpimg.imread(file_url)

    gray_image=rgb2gray(img_rgb)
    smoothed_img = convolve(gray_image, gaussian_filter(5,1.4))
    gradients,theta=filter_sobel(smoothed_img)
    non_max_image=non_max_suppression(gradients,theta)
    dbl_thr_image=double_threshold(non_max_image)
    canny_image=edge_hysteresis(dbl_thr_image)

    # In[10]:


    #dimensions of image
    h,w=gray_image.shape


    # # 2 Hough Transform

    # In[53]:

    

    def hough_transform(img):
        n_theta=181
        n_rho=181
        M,N=img.shape
        M_half,N_half=M//2,N//2
        #
        d=np.sqrt(np.square(M)+np.square(N))
        d_theta=181/n_theta
        d_rho=(2*d)/n_rho

        #
        thetas=np.arange(0,181,step=d_theta)
        rhos=np.arange(-d,d,step=d_rho)

        #
        rads=np.deg2rad(thetas)
        cos_theta=np.cos(rads)
        sin_theta=np.sin(rads)

        #
        ln=len(rhos)  
        hough_space=np.zeros((ln,ln),dtype=np.int32)

        for y in range(M):
            for x in range(N):
                if img[y][x]!=0:
                    edge=[y-M_half,x-N_half]
                    for t_i in range(len(thetas)):
                        rho=(edge[1]*cos_theta[t_i]+edge[0]*sin_theta[t_i])
                        rho_idx=np.argmin(np.abs(rhos-rho))
                        hough_space[rho_idx][t_i]+=1

        return hough_space,rhos,thetas
    hough_space,rho_h,theta_h=hough_transform(canny_image)

    hough_edges=[]
    def get_hough_edges(hough_space,rho_h,theta_h):
        mx_val=hough_space.max()
        Q=3
        threshold=math.floor(mx_val/Q)
        for t_i in range(0,len(theta_h)):
            for r_i in range((len(rho_h))):
                val=hough_space[t_i][r_i]
                if(val<threshold or rho_h[r_i]==0):
                    #Filter Rho==0 (intercept=0)
                    hough_space[t_i][r_i]=0
                else:
                    hough_edge=HoughEdge(theta_h[t_i],rho_h[r_i],val)
                    is_new_corner=True
                    for i in range(len(hough_edges)):
                        if(abs(hough_edges[i].angle-theta_h[t_i])<20
                          and abs(hough_edges[i].rho-rho_h[r_i])<100):
                            is_new_corner=False
                            if val >hough_edges[i].val:
                                hough_edges[i]=hough_edge
                                break
                    ####
                    if is_new_corner:
                        hough_edges.append(hough_edge)
        #######
        #Filter some strong edges
        if(len(hough_edges))>4:
            hough_edges.sort(key=operator.attrgetter('val'))
            while (len(hough_edges)>5):
                hough_edges.pop()
            hough_edges.sort(key=operator.attrgetter('angle'))
            if(len(hough_edges)==5):
                d_angle=abs(hough_edges[0].angle-hough_edges[1].angle)
                d_angle1=abs(hough_edges[1].angle-hough_edges[2].angle)
                d_angle2=abs(hough_edges[2].angle-hough_edges[3].angle)
                d_angle3=abs(hough_edges[3].angle-hough_edges[4].angle)
                diff=2

                #find four sides
                if(abs(d_angle-d_angle1)>diff):
                    if(d_angle < d_angle1):
                        if(abs(d_angle2-d_angle3)>diff):
                            if(d_angle2<d_angle3):
                                del hough_edges[4]
                            else:
                                del hough_edges[2]
                        else:
                            if((hough_edges[2].val<hough_edges[3].val) and
                               (hough_edges[2].val<hough_edges[4].val)):
                                del hough_edges[2]

                            elif (hough_edges[3].val < hough_edges[4].val):
                                del hough_edges[3]
                            else:
                                del hough_edges[4]
                    else:
                        del hough_edges[0]
                else:
                    if((hough_edges[0].val < hough_edges[1].val)and
                       (hough_edges[0].val < hough_edges[2].val)):
                        del hough_edges[0]

                    elif ((hough_edges[1].val < hough_edges[2].val)):
                        del hough_edges[1]
                    else:
                        del hough_edges[2] 
    get_hough_edges(hough_space,rho_h,theta_h)

    #Transform the points in hough spacce to lines in parameter space

    lines=[]
    def getLines():
        for i in range(len(hough_edges)):
            if(int(hough_edges[i].angle-180)==0):
                #when perpendicular to x-axis,dist=rho
                l1=Line(0,0,hough_edges[i].rho)
                lines.append(l1)
            theta=1.0*(hough_edges[i].angle-180)*np.pi/180.0
            m=-np.cos(theta)/np.sin(theta)
            b=(-1.0*hough_edges[i].rho)/np.sin(theta)
            lines.append(Line(m,b))
    getLines()

    #Get corners of the required image
    corners=[]
    def get_corners():
        x,y=0,0
        for i in range(len(lines)):
            for j in range(len(lines)):
                if((j==i) or (lines[i].end_point>=2)or
                  (lines[i].dist > 0) and lines[j].dist >0 ):
                    continue
                m0=lines[i].m
                b0=lines[i].b
                m1=lines[j].m
                b1=lines[j].b
                #Line i vertical
                if(lines[i].dist>0):
                    x=lines[i].dist
                    y=m1*x + b1
                    x=int(x)
                    y=int(y)
                #Line j vertical    
                elif lines[j].dist >0:
                    x=lines[j].dist
                    y=m0*x + b0
                    x=int(x)
                    y=int(y)
                else:
                    #not vertical
                    if((m0-m1)!=0):
                        _x=(b1-b0)/(m0-m1)
                        x=int(_x)
                        y=(m0 * _x +b0 + m1*_x+b1)/2
                        x=int(x)
                        y=int(y)
                D=20
                #IF intersection is out of bound
                #of an image
                #In such case we will set the intersectiong
                #point as border

                if (x>=0-D and x<h+D and y>=0-D and y< w+D):
                    if x<0:
                        x=0
                    elif x>=h:
                        x=h-1
                    if y<0:
                        y=0
                    elif y>=w:
                        y=w-1

                    if(lines[i].end_point==0):
                        lines[i].x0=x
                        lines[i].y0=y
                    elif (lines[i].end_point==1):
                        lines[i].x1=x
                        lines[i].y1=y
                if x<0:
                    x=0
                elif x>=h:
                    x=h-1
                if y<0:
                    y=0
                elif y>=w:
                    y=w-1
                corners.append(Point(x,y))
                lines[i].end_point+=1
    get_corners()

    corners_dict={}
    #dictionary to store id's of corners values
    for i in corners:
        corners_dict[id(i)]=i
    
    ordered_corners=[]
    def order_corners():
        tmp=[]
        for i in corners:
            dst=(i.x*i.x) + (i.y*i.y)
            tmp.append([dst,id(i),i.x,i.y])


        tmp.sort()    
        tmp_corners=[]
        for i in tmp:
            tmp_corners.append(corners_dict[i[1]])

        for i in range(0,len(tmp_corners),2):
            pt=Point(tmp_corners[i].x,tmp_corners[i].y)
            ordered_corners.append(pt)

        x1 = ordered_corners[0].x
        y1 = ordered_corners[0].y
        x2 = ordered_corners[1].x
        y2 = ordered_corners[1].y
        x3 = ordered_corners[2].x
        y3 = ordered_corners[2].y 
        x4 = ordered_corners[3].x
        y4 = ordered_corners[3].y
        SHIFT=5;

        if(gray_image[x1][y1]<125):
            x1+=SHIFT
            y1+=SHIFT
        if(gray_image[x2][y2]<125):
            x2-=SHIFT
            y2+=SHIFT
        if(gray_image[x3][y3]<125):
            x3+=SHIFT
            y3-=SHIFT
        if(gray_image[x4][y4]<125):
            x4-=SHIFT
            y4-=SHIFT

        if((w>h)or (x1>x2)):
            x1,x2=x2,x1
            y1,y2=y2,y1
            x3,x4=x4,x3
            y3,y4=y4,y3

        ordered_corners[0].x=x1
        ordered_corners[0].y=y1
        ordered_corners[1].x=x2
        ordered_corners[1].y=y2
        ordered_corners[2].x=x3 
        ordered_corners[2].y=y3
        ordered_corners[3].x=x4 
        ordered_corners[3].y=y4      


    order_corners()

    x1 = ordered_corners[0].x
    y1 = ordered_corners[0].y
    x2 = ordered_corners[1].x
    y2 = ordered_corners[1].y
    x3 = ordered_corners[2].x
    y3 = ordered_corners[2].y 
    x4 = ordered_corners[3].x
    y4 = ordered_corners[3].y


    def perspective_transform():    
        u1,v1,u2,v2,u3,v3,u4,v4=0,0,h-1,0,0,w-1,h-1,w-1
        UV=np.array([u1,v1,u2,v2,u3,v3,u4,v4]).reshape(-1,1)
        r1=[x1, y1, 1, 0,  0,  0, -u1*x1, -u1*y1]
        r2=[0,  0,  0, x1, y1, 1, -v1*x1, -v1*y1]
        r3=[x2, y2, 1, 0,  0,  0, -u2*x2, -u2*y2]
        r4=[0,  0,  0, x2, y2, 1, -v2*x2, -v2*y2]
        r5=[x3, y3, 1, 0,  0,  0, -u3*x3, -u3*y3]
        r6=[0,  0,  0, x3, y3, 1, -v3*x3, -v3*y3]
        r7=[x4, y4, 1, 0,  0,  0, -u4*x4, -u4*y4]
        r8=[0,  0,  0, x4, y4, 1, -v4*x4, -v4*y4]
        A=np.array([r1,r2,r3,r4,r5,r6,r7,r8]).reshape(8,-1)

        M=np.matmul(np.linalg.inv(A),UV)
        a=M[0,0]
        b=M[1,0]
        c=M[2,0]
        d=M[3,0]
        e=M[4,0]
        f=M[5,0]
        m=M[6,0]
        l=M[7,0]
        Z=[a,b,c,d,e,f,m,l]
        return Z

    Z=perspective_transform()

    a,b,c,d,e,f,m,l=Z

    rh,rw,rc=img_rgb.shape
    final = np.ones([rh,rw,rc],dtype=np.uint8)*255

    def get_x_transform_inv(u,v):
        return ((c - u)*(v*l - e) - (f - v)*(u*l - b))/((u*m - a)*(v*l - e) - (v*m - d)*(u*l - b))

    def get_y_transform_inv(u,v):
        return ((c - u)*(v*m - d) - (f - v)*(u*m - a))/((u*l - b)*(v*m - d) - (v*l - e)*(u*m - a))


    def bilinear_interpolate(x,y,c):
        i=math.floor(x)
        j=math.floor(y)
        a=x-i
        b=y-j
        return (1 - a)*(1 - b)*img_rgb[i, j, c] + a*(1 - b)*img_rgb[i + 1, j, c]+ (1 - a)*b*img_rgb[i, j + 1, c] + a*b*img_rgb[i + 1, j + 1, c]
    
    def reverse_map():
        for i in range(rh):
            for j in range(rw):
                for c in range(rc):
                    x=get_x_transform_inv(i,j)
                    y=get_y_transform_inv(i,j)
                    if(x >= 0 and y >= 0 and x + 1 < rh and y + 1 < rw):
                        final[i, j, c] = bilinear_interpolate(x, y, c)
        return final

    fnl_image=reverse_map()
    
    outfile='static/uploads/'+'out'+filename
    matplotlib.image.imsave(outfile, fnl_image)
    
    img_url=url_for('static', filename='uploads/' +'out'+filename)
    return redirect(img_url, code=301)

if __name__ == "__main__":
    app.run(port=5001)