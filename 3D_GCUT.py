# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 02 14:55:23 2016

"""

from scipy import misc, ndimage
import numpy as np
import maxflow
import math
from PyQt4 import QtGui, QtCore
import matplotlib.pyplot as plt

from PyQt4.QtCore import *
from PyQt4.QtGui import *

import sys, os
from PIL import Image, ImageQt
from scipy.interpolate import interp1d as i1
import skimage.io

"""converts grayscale image to rgb image"""
def to_rgb(im):
    w,h = im.shape
    ret= np.empty((w,h,3), dtype=np.uint8)
    ret[:,:,2]=ret[:,:,1]=ret[:,:,0]=im
    return ret

"""interpolates the image. Current Z-Resolution is fixed at 3"""
def interp_img(img, z_scale):
    z=np.linspace(0, img.shape[0], img.shape[0])
    z_new= np.linspace(0, img.shape[0], int(img.shape[0]*z_scale))   
    sweg=i1(z, img, axis=0)(z_new).astype('uint8')
    return sweg

  
def polar2cart(r, theta,phi, center):

    z = r  * np.cos(phi) + center[0]
    x = r  * np.sin(phi) * np.cos(theta) + center[2]
    y = r  * np.sin(phi) * np.sin(theta) + center[1]
    return z, x, y


"""spherical transform of cartesian image"""
def img2polar(img, center, final_radius, initial_radius = None, theta_width = 250, phi_width=250):

    if initial_radius is None:
        initial_radius = 0

    theta , R, phi = np.meshgrid(np.linspace(0, 2*np.pi, theta_width), np.arange(initial_radius, final_radius),
    np.linspace(0, np.pi, phi_width))

    Zcart, Xcart, Ycart = polar2cart(R, theta, phi, center)

    Zcart=  Zcart.astype(int)
    Xcart = Xcart.astype(int)
    Ycart = Ycart.astype(int)


    
    polar_img = img[Zcart, Xcart, Ycart]
    polar_img=  np.reshape(polar_img, (final_radius-initial_radius, theta_width, phi_width))
   
    return polar_img
    

"""Vectorized Cartesian Transform. This function avoids loops to increase speed"""
def img2cart(img, polar_img, center, radius, theta_width, phi_width):
    x,z,y= np.meshgrid(np.arange(0, img.shape[1]), np.arange(0, img.shape[0]), np.arange(0, img.shape[2]))
    rad= np.sqrt((center[0]-z)**2+ (center[1]-x)**2 + (center[2]-y)**2)
    
    theta= np.zeros(img.shape)
    th_angs= np.arctan2(center[2]-y, x-center[1])
    th_pos= th_angs*(th_angs>=0)
    th_negs= (th_angs*(th_angs<0))

    th_neg_nzeros= th_negs!=0
    th_negs= (th_negs+2*math.pi)*th_neg_nzeros



    theta= th_pos + th_negs
    theta= theta_width- theta_width*theta/(2*math.pi)-1

    phi= np.arccos((z-center[0]).astype('float')/(rad+1))
    phi= phi_width*phi/(math.pi)-1
    
    sphere= rad<radius
    rad*=sphere
    theta*=sphere
    phi*=sphere
    
    background= polar_img[rad.astype('int'), theta.astype('int'), phi.astype('int')]*sphere
    return background



def padImage(image, padList):
    
    """returns the padded image, along with the adjusted center"""

    #pad along far x:<---->
    padFarX= np.zeros((image.shape[0], image.shape[1], padList[0]))
    image= np.concatenate((image, padFarX), axis=2)

    #pad along far y
    padFarY= np.zeros((image.shape[0], padList[1], image.shape[2]))
    image= np.concatenate((image, padFarY), axis=1)

    #pad along far z
    padFarZ= np.zeros((padList[2], image.shape[1], image.shape[2]))
    image= np.concatenate((image, padFarZ), axis=0)

    #pad along close x, adjust center
    padCloseX= np.zeros((image.shape[0], image.shape[1], padList[3]))
    image= np.concatenate((padCloseX, image), axis=2)

    #pad along close y adjust center
    padCloseY= np.zeros((image.shape[0], padList[4], image.shape[2]))
    image= np.concatenate((padCloseY, image), axis=1)

    #pad along close z, adjust center
    padCloseZ= np.zeros((padList[5], image.shape[1], image.shape[2]))
    image= np.concatenate((padCloseZ, image), axis=0)


    #print "PADDED IMAGE SHAPE: " + str(image.shape)
    return image
        
        
def shearImage(image, padList):
    """shears an image based on padding specified by padList"""
    #shear along far x, y, and z
    image= image[:,:,0: image.shape[2]-padList[0]]
    image= image[:,0:image.shape[1]-padList[1],:]
    image= image[0:image.shape[0]-padList[2],:,:]

    #shear along close x, y, and z

    image= image[:,:,padList[3]:image.shape[2]]
    image= image[:,padList[4]:image.shape[1],:]
    image= image[padList[5]:image.shape[0],:,:]
     
    return image

def padCenter(center, padList):
    """adjust the center based on the padding"""
    new_center= np.array([center[0], center[1], center[2]])
    new_center[0]+=padList[5]
    new_center[1]+=padList[3]
    new_center[2]+=padList[4]

    return new_center

def graphCut(img, center, radius, temp, edge, count, editPoints, padList, theta_width, phi_width):
    
    """outputs two images. The first image shows the segmented object in white against a black background.
    The second image delineates the edge of the segmented image. Increase th_div and phi_div in their respective
    spinboxes for more accurate segmentation"""


    """Important note. The labeled image is referred to as temp, or self.temp in the interface.
    This stands for template. The previously labled image is fed back into the graphcut"""
    
    """create polar images and cost arrays"""
    
    print "RUNNING GRAPHCUT!"
    img= padImage(img, padList)
    temp= padImage(temp, padList)
    edge= padImage(edge, padList)
    center= padCenter(center, padList)
    
    polar_img= img2polar(img, center, radius, theta_width=theta_width, phi_width=phi_width)

   
        
    polar_grad, y, x = np.gradient(np.array(polar_img, dtype='float'))
    """Lockett 100416 replacement line below to not use gradient when the image has a surface label"""
    """polar_grad = -1 * np.array(polar_img, dtype='float')"""
    
                 
    polar_cost = -1 * np.ones(polar_img.shape)
    for r in range(1,radius):
            polar_cost[r]= polar_grad[r]-polar_grad[r-1]

 
    
    """
    flip the cost image upside down. This is so that the base set is at the bottom of the array
    since the graphcut cuts from top to bottom, this inversion is necessary.
    """
    polar_cost_inv=polar_cost[::-1,:,:]

    print "CONSTRUCTING GRAPH EDGES... "
    
    """construct the graph using PyMaxFlow"""
    g=maxflow.GraphFloat()
    nodeids=g.add_grid_nodes(polar_img.shape)
    structure=np.zeros((3,3,3))
    structure[2]= np.array([[0,10000,0],[10000, 10000, 10000],[0, 10000, 0]])
    g.add_grid_edges(nodeids, structure=structure, symmetric=False)

    
    """convert the previously labeled image (temp) into a polar transform image. Take the labels and
    give them high cost edge weights so the segmentation avoids previously labeled objects"""
    polar_lbl_img= img2polar(temp, center, radius, theta_width=theta_width, phi_width=phi_width)
    polar_lbl_img_inv= polar_lbl_img[::-1,:]
    
    lbl_caps= polar_lbl_img_inv>0
    self_caps= (polar_lbl_img_inv==count)
    lbl_caps-=self_caps
    lbl_source_caps= np.zeros(lbl_caps.shape)
    lbl_sink_caps= lbl_caps*10000
    g.add_grid_tedges(nodeids, lbl_source_caps, lbl_sink_caps)
   
    structure2= 10000*np.array([[0,0,0],[0,0,1],[0,1,0]])
    g.add_grid_edges(nodeids[radius-1], structure=structure2, symmetric=True)

    """add terminal edges using two arrays whose elemnts are the costs of the edges from the source and to the
    sink"""
    print "CONSTRUCTING GRAPH TEDGES..."
    sinkcaps= polar_cost_inv * (polar_cost_inv>=0)
    sourcecaps = -1 * polar_cost_inv * (polar_cost_inv<0)
    g.add_grid_tedges(nodeids, sourcecaps, sinkcaps)

  

    
    """accounts for edit points. Takes every point in the edit point list, converts it to its spherical coordinate, and adds high cost
    edges in the column of that edit point inverts the x and y coordinates of the center"""
    center= np.array((center[0], center[2], center[1]))
    if len(editPoints)!=0:
        for coords in editPoints:

            
            rad= math.sqrt((center[0]-coords[0])**2+ (center[1]-coords[2])**2 + (center[2]-coords[1])**2)            
            theta= math.atan2(center[2]-coords[1], coords[2]-center[1])
            print str((coords[0]-center[0])/(rad+1))
            phi=math.acos(float(coords[0]-center[0])/(rad+1))
            if theta<0:
                theta=2*math.pi+ theta
            theta= theta_width- theta_width*theta/(2*math.pi)-1
            phi= phi_width*phi/(math.pi)-1
            rad= radius- rad
            print "POLAR COORDS: " + str((rad, theta, phi))

            for r in range(0, radius):
                if r<=rad:
                    g.add_tedge(nodeids[r, theta, phi], 0, 10000)
                                    
                else:
                    g.add_tedge(nodeids[r, theta, phi], 10000, 0)         




    print "CUTTING GRAPH..."
    g.maxflow()

    """s-t mincut of graph. This is converted to cartesian coordinates with the function img2cart. The
    images are also closed to eliminate spotty areas"""
    
    print "STARTING CARTESIAN TRANSFORM..."
    polar_img_seg= np.invert(g.get_grid_segments(nodeids)[::-1,:,:])

    
    edge_img= np.zeros(img.shape)
    seg_img= ndimage.binary_closing(img2cart(img, polar_img_seg, center, radius, theta_width, phi_width))
    
    
    """create an edge image of the segmented object"""
    strel=np.ones((3,3,3))
    erode_img=ndimage.binary_erosion(seg_img, strel)
    edge_img=np.logical_xor(seg_img, erode_img)
    

    """shears the segmentation image and edge if padding was applied"""
    

    """add the object back on to the template image (and the edge image back on the template edge)
    If there was an editpoint involved, remove the previous segmentation of that object and add back
    on the edited object"""
    if len(editPoints)!=0:
        del_img= (temp==count)*count
        temp-=del_img

        del_edge_img= (edge==count)*count
        edge-= del_edge_img


    temp+=seg_img*count
    edge+=edge_img*count

    temp= shearImage(temp, padList)
    edge= shearImage(edge, padList)
    
    

    print "FINISHED!"
    
    return temp, edge

    

 

class Interface(QtGui.QWidget):
    def __init__(self):

	super(Interface, self).__init__()
	self.initUI()

    def initUI(self):

        """set the size of the interface, and create an empty default image for display purposes.Display
        these default images on lbl and lbl2 using pixmaps. Initialize a counter variable that tracks how
        many objects were segmented."""
        
        self.setGeometry(100,100,1500,1400)
        self.count=0
        self.editPoints= []
        self.center=np.array([])
        self.padList=np.array([])
        self.path= ''

        self.curRadius=30
        self.theta_div= 32
        self.phi_div=32
        self.zinterp=3

        
        self.img= np.zeros((25, 256, 256))
        self.edge= interp_img(np.zeros(self.img.shape), self.zinterp)
        self.temp= interp_img(np.zeros(self.img.shape), self.zinterp)
        self.shrink= np.zeros(self.img.shape)
    
       
        self.z_stack=self.img.shape[0]/2
        self.y_stack= self.img.shape[1]/2
        self.x_stack= self.img.shape[2]/2

        self.dispedge = to_rgb(self.img[self.z_stack])
        self.y_dispedge= to_rgb(self.img[:,self.y_stack,:])
        self.x_dispedge= to_rgb(self.img[:,:,self.x_stack])

        
        
        self.pixmap=QtGui.QPixmap.fromImage(ImageQt.ImageQt(misc.toimage(self.img[self.z_stack]))).scaled(250,250)
        self.pixmap2=QtGui.QPixmap.fromImage(ImageQt.ImageQt(misc.toimage(self.img[self.z_stack]))).scaled(250,250)
        self.pixmap3=QtGui.QPixmap.fromImage(ImageQt.ImageQt(misc.toimage(self.img[:,100,:]))).scaled(250,250)
        self.pixmap4=QtGui.QPixmap.fromImage(ImageQt.ImageQt(misc.toimage(self.img[:,100,:]))).scaled(250,250)
        
        self.pixmap5=QtGui.QPixmap.fromImage(ImageQt.ImageQt(misc.toimage(self.img[:,:,100]))).scaled(250,250)
        self.pixmap6=QtGui.QPixmap.fromImage(ImageQt.ImageQt(misc.toimage(self.img[:,:,100]))).scaled(250,250)


        self.lbl = QtGui.QLabel(self)
        self.lbl.setPixmap(self.pixmap)
        self.lbl.move(100,200)
        self.lbl.installEventFilter(self)
        self.lblString=str(self.lbl)

        self.lbl2= QtGui.QLabel(self)
        self.lbl2.setPixmap(self.pixmap2)   
        self.lbl2.move(600,200)
        self.lbl2.installEventFilter(self)
        self.lbl2String=str(self.lbl2)
        #self.lbl2.mousePressEvent = self.editAndDelete
        
        self.lbl3 = QtGui.QLabel(self)
        self.lbl3.setPixmap(self.pixmap3)
        self.lbl3.move(100,500)
        self.lbl3.installEventFilter(self)
        self.lbl3String=str(self.lbl3)

        #self.lbl3.mousePressEvent=self.chooseObject
        
        self.lbl4= QtGui.QLabel(self)
        self.lbl4.setPixmap(self.pixmap4)   
        self.lbl4.move(600,500)
        self.lbl4.installEventFilter(self)
        self.lbl4String=str(self.lbl4)
        
        self.lbl5= QtGui.QLabel(self)
        self.lbl5.setPixmap(self.pixmap5)   
        self.lbl5.move(100,800)
        self.lbl5.installEventFilter(self)
        self.lbl5String=str(self.lbl5)
        
        self.lbl6= QtGui.QLabel(self)
        self.lbl6.setPixmap(self.pixmap5)   
        self.lbl6.move(600,800)
        self.lbl6.installEventFilter(self)
        self.lbl6String=str(self.lbl6)

        self.z_stack_lbl=QtGui.QLabel(str(self.z_stack+1) + '/' + str(self.img.shape[0]), self)
        self.y_stack_lbl=QtGui.QLabel(str(self.y_stack+1) + '/' + str(self.img.shape[1]), self)
        self.x_stack_lbl=QtGui.QLabel(str(self.x_stack+1) + '/' + str(self.img.shape[2]), self)

        self.z_stack_lbl.move(410,320)
        self.y_stack_lbl.move(405,620)
        self.x_stack_lbl.move(405,920)

        """Create boxes that allow the users to edit the radius, theta divisions and
        phi divisions'"""
        self.edit_rad= QtGui.QSpinBox(self)
        self.edit_rad.setGeometry(950,200,100,60)
        self.edit_rad.setValue(30)
        self.edit_rad.valueChanged.connect(self.changeRadius)
        self.edit_rad.setRange(10, 500)
        
      
        self.edit_th_divs= QtGui.QSpinBox(self)
        self.edit_th_divs.setGeometry(950, 350, 100, 60)
        self.edit_th_divs.setValue(self.theta_div)
        self.edit_th_divs.valueChanged.connect(self.change_th_divs)
        self.edit_th_divs.setRange(32, 1000)


        self.edit_phi_divs= QtGui.QSpinBox(self)
        self.edit_phi_divs.setGeometry(950, 500, 100, 60)
        self.edit_phi_divs.setValue(self.phi_div)
        self.edit_phi_divs.valueChanged.connect(self.change_phi_divs)
        self.edit_phi_divs.setRange(32, 1000)
        
        self.edit_z_interp= QtGui.QSpinBox(self)
        self.edit_z_interp.setGeometry(950, 650, 100, 60)
        self.edit_z_interp.setValue(self.zinterp)
        self.edit_z_interp.valueChanged.connect(self.change_z_interp)
        self.edit_z_interp.setRange(1,5)


        """creates up, down, top, and bottom buttons that allow the user
        to scroll efficiently through the image stacks in the Z Direction."""
        self.upbtn= QtGui.QPushButton("up", self)
        self.upbtn.setGeometry(400, 263, 50,50)
        self.upbtn.clicked.connect(self.scrollUp)
        self.upbtn.setAutoRepeat(True)
        self.upbtn.setAutoRepeatInterval(75)
        
        self.downbtn= QtGui.QPushButton("down", self)
        self.downbtn.setGeometry(400,358, 75,50)
        self.downbtn.clicked.connect(self.scrollDown)
        self.downbtn.setAutoRepeat(True)
        self.downbtn.setAutoRepeatInterval(75)

        self.topbtn= QtGui.QPushButton("top", self)
        self.topbtn.setGeometry(400, 200, 50,50)
        self.topbtn.clicked.connect(self.topStack)

        self.botbtn= QtGui.QPushButton("bot", self)
        self.botbtn.setGeometry(400, 420, 50,50)
        self.botbtn.clicked.connect(self.botStack)
        
        """add buttons that scroll through the xz slices"""
        self.xz_upbtn= QtGui.QPushButton("up_y", self)
        self.xz_upbtn.setGeometry(400, 563, 75,50)
        self.xz_upbtn.setAutoRepeat(True)
        self.xz_upbtn.clicked.connect(self.scrollUp_y)
        self.xz_upbtn.setAutoRepeatInterval(75)
        
        self.xz_downbtn= QtGui.QPushButton("down_y", self)
        self.xz_downbtn.setGeometry(400, 658, 100,50)
        self.xz_downbtn.clicked.connect(self.scrollDown_y)
        self.xz_downbtn.setAutoRepeat(True)
        self.xz_downbtn.setAutoRepeatInterval(75)
        
        self.xz_topbtn= QtGui.QPushButton("top_y", self)
        self.xz_topbtn.setGeometry(400, 500, 85,50)
        self.xz_topbtn.clicked.connect(self.topStack_y)
       
        self.xz_botbtn= QtGui.QPushButton("bot_y", self)
        self.xz_botbtn.setGeometry(400, 720, 85,50)
        self.xz_botbtn.clicked.connect(self.botStack_y)
        
        """add buttons that scroll through the yz slices"""
        self.yz_upbtn= QtGui.QPushButton("up_x", self)
        self.yz_upbtn.setGeometry(400, 863, 75,50)
        self.yz_upbtn.setAutoRepeat(True)
        self.yz_upbtn.clicked.connect(self.scrollUp_x)
        self.yz_upbtn.setAutoRepeatInterval(75)
        
        self.yz_downbtn= QtGui.QPushButton("down_x", self)
        self.yz_downbtn.setGeometry(400, 958, 100,50)
        self.yz_downbtn.clicked.connect(self.scrollDown_x)
        self.yz_downbtn.setAutoRepeat(True)
        self.yz_downbtn.setAutoRepeatInterval(75)
        
        self.yz_topbtn= QtGui.QPushButton("top_x", self)
        self.yz_topbtn.setGeometry(400, 800, 85,50)
        self.yz_topbtn.clicked.connect(self.topStack_x)
    
        self.yz_botbtn= QtGui.QPushButton("bot_x", self)
        self.yz_botbtn.setGeometry(400, 1020, 85,50)
        self.yz_botbtn.clicked.connect(self.botStack_x)
       
        """Adds browse and save buttons"""
        self.browsebtn= QtGui.QPushButton("Browse for an image", self)
        self.browsebtn.setGeometry(100,1100, 270,80)
        self.savebtn = QtGui.QPushButton("Save Segmentation", self)
        self.savebtn.setGeometry(500,1100, 250,80)
        
        self.loadbtn= QtGui.QPushButton("Load previous images", self)
        self.loadbtn.setGeometry(850,1100, 270,80)

                
        self.savebtn.clicked.connect(self.saveFile)
        self.browsebtn.clicked.connect(self.getFile)
        self.loadbtn.clicked.connect(self.loadFile)
        
        


        """adds text labels and set their locations in the interface"""
        qf=QtGui.QFont("Arial", 20)
        qfbold=QtGui.QFont("Arial", 20, QtGui.QFont.Bold)
        self.txtlbl1=QtGui.QLabel('Original Image', self)
        self.txtlbl2=QtGui.QLabel('Segmented Image', self)
        self.txtlbl3=QtGui.QLabel('Welcome to OMAL GraphCut Segmentation Interface!', self)
        self.txtlbl4=QtGui.QLabel('Change default radius', self)
        self.txtlbl5=QtGui.QLabel('Change theta divisions', self)
        self.txtlbl6=QtGui.QLabel('Change phi divisions', self)
        self.txtlbl7=QtGui.QLabel('Change z interpolation', self)
        self.txtlbl8=QtGui.QLabel('Dimensions: ' + str(self.img.shape[0]) + 'x ' + str(self.img.shape[1]) + 'x' +str(self.img.shape[2]), self)
	


        self.txtlbl1.setFont(qf)
        self.txtlbl2.setFont(qf)
        self.txtlbl1.move(100,115)
        self.txtlbl2.move(600,115)
        self.txtlbl3.move(100,25)
        self.txtlbl3.setFont(qfbold)
        self.txtlbl4.move(875,275)
        self.txtlbl5.move(875,425)
        self.txtlbl6.move(875, 575)
        self.txtlbl7.move(875,725)
        #self.txtlbl6.setFont(QtGui.QFont("Arial", 12))

        self.txtlbl8.move(875,850)
        self.txtlbl8.setFont(QtGui.QFont("Arial", 12))
        
        
   
        
    """eventFilter method that is installed by the label (lb1, lbl2, etc)
    this method will recognize which label was pressed, modify the coordinates
    pressed into a center point and call chooseObject and editAndDelte accordingly"""
    
    def eventFilter(self, source, event):
        """retreives which label was pressed, creates the respective x,y, and z coordinates
        in the image, and calls the appropriate function (deleteObject, chooseObject, and editObject)"""        
        if event.type() == QtCore.QEvent.MouseButtonPress:
             
            if str(source)==self.lblString:
                x= event.pos().x()*self.img.shape[2]/250
                y= event.pos().y()*self.img.shape[1]/250
                z= self.z_stack
                self.chooseObject(x,y,z)
                self.calibrateStacks(x,y,z)

            
            elif str(source)==self.lbl2String:
                x= event.pos().x()*self.img.shape[2]/250
                y= event.pos().y()*self.img.shape[1]/250
                z= self.z_stack
                
                if event.button()==QtCore.Qt.LeftButton: 
                    self.editObject(x,y,z)
                    self.calibrateStacks(x,y,z)

                elif event.button()==QtCore.Qt.RightButton:
                    self.deleteObject(x,y,z)
                
            elif str(source)==self.lbl3String:
                x=event.pos().x()*self.img.shape[2]/250
                y=self.y_stack
                z=event.pos().y()*self.img.shape[0]/250  
                self.chooseObject(x,y,z)
                self.calibrateStacks(x,y,z)

                
            elif str(source)==self.lbl4String:
                x=event.pos().x()*self.img.shape[2]/250
                y=self.y_stack
                z=event.pos().y()*self.img.shape[0]/250 
                
                if event.button()==QtCore.Qt.LeftButton: 
                    self.editObject(x,y,z)
                    self.calibrateStacks(x,y,z)

                elif event.button()==QtCore.Qt.RightButton:
                    self.deleteObject(x,y,z)
                
                
            elif str(source)==self.lbl5String:
                x=self.x_stack                
                y=event.pos().x()*self.img.shape[1]/250
                z=event.pos().y()*self.img.shape[0]/250  
                self.chooseObject(x,y,z)
                self.calibrateStacks(x,y,z)

            
            elif str(source)==self.lbl6String:
                x=self.x_stack                
                y=event.pos().x()*self.img.shape[1]/250
                z=event.pos().y()*self.img.shape[0]/250 
                
                if event.button()==QtCore.Qt.LeftButton: 
                    self.editObject(x,y,z)
                    self.calibrateStacks(x,y,z)

                elif event.button()==QtCore.Qt.RightButton:
                    self.deleteObject(x,y,z)   
            else:
                pass
            

                
        return super(Interface, self).eventFilter(source, event)
    
    
    def calibrateStacks(self, x_stack, y_stack, z_stack):
        """automatically adjust the display when an object is segmented. Allows for viewing of the object
        from all three orthagonal views"""
        self.x_stack= x_stack
        self.y_stack=y_stack
        self.z_stack=z_stack
        
        self.resetImages()
        
        
        
        
    def chooseObject(self, x, y, z):
        

        """This method converts a user click on the original image and creates the red boundary"""

        """deletes the editPoints list. This must be reset every time a new object
        is clicked on so that the prevous editpoints from another segmentation do not
        interfere with the current object's segmentation"""
        
        del self.editPoints[:]
        
        self.dispedge = to_rgb(self.img[self.z_stack])
        
        #print "SENDER LABEL: " + str(sender.text()
        self.center= np.array((z*self.zinterp, x, y))

        xpix=self.img.shape[2]-1
        ypix= self.img.shape[1]-1
        zpix= self.img.shape[0]*self.zinterp-1
        

        #currently padding all sides by 50 for now.

        self.radius= self.curRadius
        self.count+=1
 
        self.padList= np.array([xpix-x, ypix-y, zpix-z*self.zinterp, x, y,z*self.zinterp])
        self.padList= self.radius-self.padList
        self.padList=(self.padList>0)*self.padList

        """perform graphcut and display on interface"""
        self.temp, self.edge= graphCut(interp_img(self.img, self.zinterp), self.center, self.radius,self.temp, self.edge, self.count, self.editPoints, self.padList, self.theta_div, self.phi_div)
        self.shrink= self.edge[0:interp_img(self.img, self.zinterp).shape[0]:self.zinterp]!=0
        
        """display the object on the xy plane"""
        self.pixmap2=self.writeEdge("xy")
        self.lbl2.setPixmap(self.pixmap2)
        
        """display the object on the xz plane"""
        self.pixmap4=self.writeEdge("xz")
        self.lbl4.setPixmap(self.pixmap4)
        
        """display the object on the yz plane"""
        self.pixmap6=self.writeEdge("yz")
        self.lbl6.setPixmap(self.pixmap6)
        
        

        
             
    def editObject(self, x, y, z):
            """edits and deletes segmented objects. Makes sure that a center has already been selected"""
            if self.center.size!=0:
                edit_center=padCenter([z*self.zinterp,x,y], self.padList)
                
                """executes when a left click is done on lbl2 (the right hand image) and inserts
                an edit point"""
                
                #print "getting edit points!..."
                self.editPoints.append(edit_center)
                print "EDITPOINTS: " + str(self.editPoints)

                
                self.temp, self.edge= graphCut(interp_img(self.img, self.zinterp),self.center,self.radius,self.temp,self.edge,self.count,self.editPoints,self.padList,self.theta_div,self.phi_div)
                self.shrink= self.edge[0:interp_img(self.img, self.zinterp).shape[0]:self.zinterp]!=0
                
                """display the image on the xy plane"""
                self.pixmap2=self.writeEdge("xy")
                self.lbl2.setPixmap(self.pixmap2)
                   
                """display the image on the xz plane"""
                self.pixmap4=self.writeEdge("xz")
                self.lbl4.setPixmap(self.pixmap4)
                   
                """display the image on the yz plane"""
                self.pixmap6=self.writeEdge("yz")
                self.lbl6.setPixmap(self.pixmap6)
    
               
     
    def deleteObject(self, x, y, z):
        lbl_del= self.temp[z*self.zinterp,y,x]
        if lbl_del!=0:
             del_img= (self.temp==lbl_del)*lbl_del
             del_edge_img=(self.edge==lbl_del)*lbl_del
             subtract_1=self.temp>lbl_del
             subtract_edge_1= self.edge>lbl_del
        
             self.temp-=del_img-subtract_1
             self.edge-=del_edge_img-subtract_edge_1
             self.count-=1
             self.shrink= self.edge[0:interp_img(self.img,self.zinterp).shape[0]:self.zinterp]!=0
           
             """display the image on the xy plane"""
             self.pixmap2=self.writeEdge("xy")
             self.lbl2.setPixmap(self.pixmap2)
           
             """display the image on the xz plane"""
             self.pixmap4=self.writeEdge("xz")
             self.lbl4.setPixmap(self.pixmap4)
           
             """display the image on the yz plane"""
             self.pixmap6=self.writeEdge("yz")
             self.lbl6.setPixmap(self.pixmap6)

    def scrollUp(self):
        """method that corresponds to the up button in initUI. Allows the user to scroll up a stack"""
        if self.z_stack>0:
            self.z_stack-=1
            self.pixmap=self.drawPixmap("xy")
            self.lbl.setPixmap(self.pixmap)
            self.pixmap2=self.writeEdge("xy")
            self.lbl2.setPixmap(self.pixmap2)

            self.z_stack_lbl.setText(str(self.z_stack+1) + '/' + str(self.img.shape[0]))
            
    def topStack(self):
        """method that corresponds to the top button in initUI. Allows the user to display the top stack"""

        self.z_stack=0
        #self.pixmap=QtGui.QPixmap.fromImage(ImageQt.ImageQt(misc.toimage(self.img[self.z_stack]))).scaled(500,500)
        self.pixmap= self.drawPixmap("xy")
        self.lbl.setPixmap(self.pixmap)

        self.pixmap2=self.writeEdge("xy")
        self.lbl2.setPixmap(self.pixmap2)
        self.z_stack_lbl.setText(str(self.z_stack+1) + '/' + str(self.img.shape[0]))


    def botStack(self):
        """method that corresponds to the bot button in initUI. Allows the user to display the bottom stack"""

        self.z_stack=self.img.shape[0]-1
        #self.pixmap=QtGui.QPixmap.fromImage(ImageQt.ImageQt(misc.toimage(self.img[self.z_stack]))).scaleds(500,500)
        self.pixmap= self.drawPixmap("xy")
        self.lbl.setPixmap(self.pixmap)

        self.pixmap2=self.writeEdge("xy")
        self.lbl2.setPixmap(self.pixmap2)
        self.z_stack_lbl.setText(str(self.z_stack+1) + '/' + str(self.img.shape[0]))
        
    def scrollDown(self):

        """method that corresponds to the down button in initUI. Allows the user to scroll down a stack"""

        if self.z_stack<self.img.shape[0]-1:
            self.z_stack+=1
            
            #self.pixmap=QtGui.QPixmap.fromImage(ImageQt.ImageQt(misc.toimage(self.img[self.z_stack]))).scaled(500,500)
            self.pixmap= self.drawPixmap("xy")
            self.lbl.setPixmap(self.pixmap)
            self.pixmap2= self.writeEdge("xy")
            self.lbl2.setPixmap(self.pixmap2)
            self.z_stack_lbl.setText(str(self.z_stack+1) + '/' + str(self.img.shape[0]))
    
     
            
    def scrollUp_y(self):
        """method that corresponds to the up button in initUI. Allows the user to scroll up a stack"""
        if self.y_stack>0:
            self.y_stack-=1
            
            self.pixmap3=self.drawPixmap("xz")
            self.lbl3.setPixmap(self.pixmap3)
            self.pixmap4= self.writeEdge("xz")
            self.lbl4.setPixmap(self.pixmap4)
            self.y_stack_lbl.setText(str(self.y_stack+1) + '/' + str(self.img.shape[1]))
            
    
    def scrollDown_y(self):
        """method that corresponds to the up button in initUI. Allows the user to scroll up a stack"""
        if self.y_stack<self.img.shape[1]-1:
            self.y_stack+=1
            self.pixmap3=self.drawPixmap("xz")           
            self.lbl3.setPixmap(self.pixmap3)
            self.pixmap4= self.writeEdge("xz")
            self.lbl4.setPixmap(self.pixmap4)
            self.y_stack_lbl.setText(str(self.y_stack+1) + '/' + str(self.img.shape[1]))

    
    def topStack_y(self):
        """method that corresponds to the up button in initUI. Allows the user to scroll up a stack"""
        self.y_stack=0
        
        self.pixmap3=self.drawPixmap("xz")           
        self.lbl3.setPixmap(self.pixmap3)
        self.pixmap4= self.writeEdge("xz")
        self.lbl4.setPixmap(self.pixmap4)
        self.y_stack_lbl.setText(str(self.y_stack+1) + '/' + str(self.img.shape[1]))


    
    def botStack_y(self):
        """method that corresponds to the up button in initUI. Allows the user to scroll up a stack"""
        self.y_stack=self.img.shape[1]-1
        
        self.pixmap3=self.drawPixmap("xz")
        self.lbl3.setPixmap(self.pixmap3)
        self.pixmap4= self.writeEdge("xz")
        self.lbl4.setPixmap(self.pixmap4)
        self.y_stack_lbl.setText(str(self.y_stack+1) + '/' + str(self.img.shape[1]))

        
    
    def scrollUp_x(self):
        """method that corresponds to the up button in initUI. Allows the user to scroll up a stack"""
        if self.x_stack>0:
            self.x_stack-=1
            
            self.pixmap5=self.drawPixmap("yz")
            self.lbl5.setPixmap(self.pixmap5) 
            self.pixmap6= self.writeEdge("yz")
            self.lbl6.setPixmap(self.pixmap6)
            self.x_stack_lbl.setText(str(self.x_stack+1) + '/' + str(self.img.shape[2]))

    
    def scrollDown_x(self):
        """method that corresponds to the up button in initUI. Allows the user to scroll up a stack"""
        if self.x_stack<self.img.shape[2]-1:
            self.x_stack+=1
            
            self.pixmap5=self.drawPixmap("yz")
            self.lbl5.setPixmap(self.pixmap5)
            self.pixmap6= self.writeEdge("yz")
            self.lbl6.setPixmap(self.pixmap6)
            self.x_stack_lbl.setText(str(self.x_stack+1) + '/' + str(self.img.shape[2]))

            
    def topStack_x(self):
        """method that corresponds to the up button in initUI. Allows the user to scroll up a stack"""
        self.x_stack=0
        
        self.pixmap5=self.drawPixmap("yz")
        self.lbl5.setPixmap(self.pixmap5)
        self.pixmap6= self.writeEdge("yz")
        self.lbl6.setPixmap(self.pixmap6)
        self.x_stack_lbl.setText(str(self.x_stack+1) + '/' + str(self.img.shape[2]))

    
    def botStack_x(self):
        """method that corresponds to the up button in initUI. Allows the user to scroll up a stack"""
        self.x_stack=self.img.shape[2]-1
        
        self.pixmap5=self.drawPixmap("yz")
        self.lbl5.setPixmap(self.pixmap5)
        self.pixmap6= self.writeEdge("yz")
        self.lbl6.setPixmap(self.pixmap6)
        self.x_stack_lbl.setText(str(self.x_stack+1) + '/' + str(self.img.shape[2]))

              
    def drawPixmap(self, stack_type):
        """draws the pixmap for all left hand side images."""
        if stack_type=="xy":
            im=Image.fromarray(to_rgb(self.img[self.z_stack]))
            image= ImageQt.ImageQt(im)
            image2= QtGui.QImage(image)
            pixmap=QtGui.QPixmap.fromImage(image2).scaled(250,250)
            return pixmap
        
        elif stack_type=="xz":
            im=Image.fromarray(to_rgb(self.img[:,self.y_stack,:]))
            image= ImageQt.ImageQt(im)
            image2= QtGui.QImage(image)
            pixmap=QtGui.QPixmap.fromImage(image2).scaled(250,250)
            return pixmap
        
        else:
            im=Image.fromarray(to_rgb(self.img[:,:,self.x_stack]))
            image= ImageQt.ImageQt(im)
            image2= QtGui.QImage(image)
            pixmap=QtGui.QPixmap.fromImage(image2).scaled(250,250)
            return pixmap
            
        #self.lbl2.setPixmap(self.pixmap2)
        
    def writeEdge(self, stack_type):
        """This function draws the red boundary of the black and white edge image (self.edge)"""

        """Set the red channel of self.dispedge equal to 255, and set the blue and green channels equal to
        zero where there is an edge pixel"""
        if stack_type=="xy":
            
            self.dispedge[:,:,0]= np.maximum(self.img[self.z_stack], self.shrink[self.z_stack]*255)
            self.dispedge[:,:,1]=np.minimum(self.img[self.z_stack], 255-self.shrink[self.z_stack]*255)
            self.dispedge[:,:,2]=self.dispedge[:,:,1]
            
           
                
            """displays the  red boundary on the pixmap by using the PIL library"""
            im=Image.fromarray(self.dispedge)
            image= ImageQt.ImageQt(im)
            image2= QtGui.QImage(image)
            pixmap=QtGui.QPixmap.fromImage(image2).scaled(250,250)
            #self.lbl2.setPixmap(self.pixmap2)
            return pixmap
            
        elif stack_type=="xz":
            self.y_dispedge[:,:,0]= np.maximum(self.img[:,self.y_stack,:], self.shrink[:,self.y_stack,:]*255)
            self.y_dispedge[:,:,1]=np.minimum(self.img[:,self.y_stack,:], 255-self.shrink[:,self.y_stack,:]*255)
            self.y_dispedge[:,:,2]=self.y_dispedge[:,:,1]
    
            """displays the  red boundary on the pixmap by using the PIL library"""
            im=Image.fromarray(self.y_dispedge)
            image= ImageQt.ImageQt(im)
            image2= QtGui.QImage(image)
            pixmap=QtGui.QPixmap.fromImage(image2).scaled(250,250)
            #self.lbl2.setPixmap(self.pixmap2)
            return pixmap
            
        else:
            self.x_dispedge[:,:,0]= np.maximum(self.img[:,:, self.x_stack], self.shrink[:,:,self.x_stack]*255)
            self.x_dispedge[:,:,1]=np.minimum(self.img[:,:,self.x_stack], 255-self.shrink[:,:,self.x_stack]*255)
            self.x_dispedge[:,:,2]=self.x_dispedge[:,:,1]
    
            """displays the  red boundary on the pixmap by using the PIL library"""
            im=Image.fromarray(self.x_dispedge)
            image= ImageQt.ImageQt(im)
            image2= QtGui.QImage(image)
            pixmap=QtGui.QPixmap.fromImage(image2).scaled(250,250)
            #self.lbl2.setPixmap(self.pixmap2)
            return pixmap
          
    def resetImages(self):
        
        self.pixmap= self.drawPixmap("xy")
        self.pixmap2=self.writeEdge("xy")
        self.pixmap3=self.drawPixmap("xz")
        self.pixmap4=self.writeEdge("xz")
        self.pixmap5=self.drawPixmap("yz")
        self.pixmap6=self.writeEdge("yz")

        self.lbl.setPixmap(self.pixmap)
        self.lbl2.setPixmap(self.pixmap2)
        self.lbl3.setPixmap(self.pixmap3)
        self.lbl4.setPixmap(self.pixmap4)
        self.lbl5.setPixmap(self.pixmap5)
        self.lbl6.setPixmap(self.pixmap6)

        self.z_stack_lbl.setText(str(self.z_stack+1) + '/' + str(self.img.shape[0]))
        self.y_stack_lbl.setText(str(self.y_stack+1) + '/' + str(self.img.shape[1]))
        self.x_stack_lbl.setText(str(self.x_stack+1) + '/' + str(self.img.shape[2]))
            
        self.txtlbl8.setText('Dimensions: ' + str(self.img.shape[0]) + 'x ' + str(self.img.shape[1]) + 'x' +str(self.img.shape[2]))
        
    def saveFile(self):
        dir_name=str(QtGui.QFileDialog.getExistingDirectory(self, "Choose Directory "))

        """gets the name of the filename without the file extension"""
        base=os.path.basename(str(self.file_name))
        file_no_ext= os.path.splitext(base)[0]

        """takes the file name without extensions and adds _lbl_img and _edge_img suffixes"""
        fname_1= dir_name+  '/' + file_no_ext +'_lbl_img.tif'
        fname_2= dir_name+ '/' + file_no_ext + '_edge_img.tif'
        
        """saves the image"""
        skimage.io.imsave(fname_1, self.temp[0:interp_img(self.img, self.zinterp).shape[0]:self.zinterp].astype('uint8'), plugin='tifffile')
        skimage.io.imsave(fname_2, (self.shrink!=0).astype('uint8'), plugin='tifffile')

    def changeRadius(self):
        """This function changes the radius when the 'change radius' spinbox is clicked or edited"""
        self.curRadius= self.edit_rad.value()
        
    def change_th_divs(self):
        """This function changes the theta divisions when the 'change theta divisions' spinbox is clicked or edited"""
        self.theta_div = self.edit_th_divs.value()

    def change_phi_divs(self):
        """This function changes the phi divisions when the 'change phi divisions' spinbox is clicked or edited"""
        self.phi_div = self.edit_phi_divs.value()
    
    def change_z_interp(self):
        self.z_interp= self.edit_z_interp.value()
        
    def loadFile(self):
        self.file_name=QtGui.QFileDialog.getOpenFileName(self, "Select Image to load", self.path, "*tif")
        temp_name= QtGui.QFileDialog.getOpenFileName(self, "Select labeled image to load", self.path, "*tif")
        if self.file_name!='' and temp_name!='':
            self.img= skimage.io.imread(str(self.file_name), plugin='tifffile')
                        
            temp_small=skimage.io.imread(str(temp_name), plugin='tifffile')
                        
            self.temp=interp_img(np.zeros(self.img.shape), self.zinterp)
            self.edge=interp_img(np.zeros(self.img.shape), self.zinterp)
            
            for label in range(1, temp_small.max()+1):
                cur_object= (temp_small==label)                
                if cur_object.sum()>0:
                    cur_object_large=interp_img(cur_object, self.zinterp)!=0
                    self.temp+=cur_object_large*label
                    
                    cur_object_large_erode= ndimage.binary_erosion(cur_object_large, np.ones((3,3,3)))
                    self.edge+=np.logical_xor(cur_object_large, cur_object_large_erode)*label
                    
            
        
            self.shrink= self.edge[0:interp_img(self.img, self.zinterp).shape[0]:self.zinterp]!=0
            self.count=temp_small.max()+1
            
            self.z_stack=self.img.shape[0]/2
            self.y_stack=self.img.shape[1]/2
            self.x_stack=self.img.shape[2]/2
            
            self.dispedge = to_rgb(self.img[self.z_stack])
            self.y_dispedge= to_rgb(self.img[:,self.y_stack,:])
            self.x_dispedge= to_rgb(self.img[:,:,self.x_stack])
            
            self.resetImages()            
           
                    


            
    
    def getFile(self):
        """gets the filename from the file directory"""
        self.file_name=QtGui.QFileDialog.getOpenFileName(self, "Open Image file", self.path, "*tif")
        if self.file_name!='':
            
            self.img= skimage.io.imread(str(self.file_name), plugin='tifffile')
            """sets self.img equal to the chosen image"""
            
            self.temp= interp_img(np.zeros(self.img.shape), self.zinterp)
            self.edge= interp_img(np.zeros(self.img.shape), self.zinterp)
            self.shrink= np.zeros(self.img.shape)
            self.count=0
            
            self.z_stack=self.img.shape[0]/2
            self.y_stack=self.img.shape[1]/2
            self.x_stack=self.img.shape[2]/2
            
            self.dispedge = to_rgb(self.img[self.z_stack])
            self.y_dispedge= to_rgb(self.img[:,self.y_stack,:])
            self.x_dispedge= to_rgb(self.img[:,:,self.x_stack])
            #self.pixmap=QtGui.QPixmap.fromImage(ImageQt.ImageQt(misc.toimage(self.img[self.z_stack]))).scaled(500,500)
            
            self.resetImages()
            


    

    
"""Launches the Interface"""
  
def main():
    app = QtGui.QApplication(sys.argv)
    interfaceWindow = Interface()
    
    """place the interface widget inside a scrollarea widget, and set an icon
    and a window title"""
    scrollArea=QtGui.QScrollArea()
    scrollArea.setWidget(interfaceWindow)
    scrollArea.setWindowTitle('3D segmentation')
    scrollArea.setWindowIcon(QtGui.QIcon('graph_icon.png')) 
    scrollArea.show()
    sys.exit(app.exec_())

if __name__=='__main__':
    main()



