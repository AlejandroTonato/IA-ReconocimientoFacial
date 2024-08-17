import PySimpleGUI as sg
import cv2
import os
import imutils
import numpy as np

sg.theme('DarkGrey')
itemsInfo = []
rostros =[]
lstPersonas = sg.Listbox(itemsInfo, size=(30, 4), font=('Arial Bold', 10), expand_y=True, enable_events=True, key='-LIST-')
lstImagenes = sg.Listbox(rostros, size=(30, 4), font=('Arial Bold', 10), expand_y=True, enable_events=True, key='-LIST1-')
layout = [
    [sg.Text('\tInteligencia Artificial ', font=('Arial Bold', 18))],
    [sg.Text('\t       Prototipo Detección de Rostros\n', font=('Arial Bold', 12))],
    [sg.Text('Persona a Identificar: '), sg.InputText(key="newPersona")],
    [sg.Button('CapturarImagen')],
    [sg.Text('****************************************************************', font=('Arial Bold', 14))],
    [sg.Button('Entrenar'), sg.Text('\t\t\t Rostros a detectar de: ', font=('Arial Bold', 10))],
    [lstPersonas,lstImagenes],
    [sg.Text('\t\t\t\t', font=('Arial Bold', 10)),sg.Button('Detectar')],
    [sg.Button('Salir')]
]
window = sg.Window('Detecccion de Rostros', layout, margins=(10, 10))

while True:
    event, values = window.read()
    # 1ro Capturar Imagenes de personas
    if event == 'CapturarImagen':
            personName = values['newPersona']
            dataPath = 'imagenes' #Cambia a la ruta donde hayas almacenado Data
            personPath = dataPath + '/' + personName
            numCamara = 0
            
            if not os.path.exists(personPath):
                print('Carpeta creada: ',personPath)
                os.makedirs(personPath)
            
            try:
                cap = cv2.VideoCapture(numCamara,cv2.CAP_DSHOW)  # 0, 1 son los índices de la cámara
                #url = 'http://192.168.100.7:4747/video'
                #cap = cv2.VideoCapture(url)
                #cap = cv2.VideoCapture('Video.mp4')
            except Exception as error:
                print('Error con algo de la cámara ' + str(error))
            faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
            count = 0
            
            while True:
                ret, frame = cap.read()
                if ret == False: break
                frame =  imutils.resize(frame, width=640)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                auxFrame = frame.copy()
            
                faces = faceClassif.detectMultiScale(gray,1.3,5)
            
                for (x,y,w,h) in faces:
                    cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
                    rostro = auxFrame[y:y+h,x:x+w]
                    rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite(personPath + '/rotro_{}.jpg'.format(count),rostro)
                    count = count + 1
                cv2.imshow('frame',frame)
            
                k =  cv2.waitKey(1)
                if k == 27 or count >= 300:
                    break            
            cap.release()
            cv2.destroyAllWindows()
    
    # 2do Entrenar con todas las imagenes capturadas (Reconocimiento Facial)
    if event == 'Entrenar':
    
        resultado = sg.popup_ok_cancel("\t ....Press Ok para entrenar!!", "",  title="Entrenar...")
        
        if resultado == "OK":
            itemsInfo.append('Leyendo las imágenes')
            itemsInfo.append('Entrenando...')
            window['-LIST-'].update(itemsInfo)
            window.refresh()
            dataPath = 'imagenes' #Cambia a la ruta donde hayas almacenado Data
            peopleList = os.listdir(dataPath)
            rostros = peopleList 
            print('Lista de personas: ', peopleList)
            
            labels = []
            facesData = []
            label = 0
            
            for nameDir in peopleList:
                personPath = dataPath + '/' + nameDir
                for fileName in os.listdir(personPath):
                    print('Caras: ', nameDir + '/' + fileName)
                    labels.append(label)
                    facesData.append(cv2.imread(personPath+'/'+fileName,0))
                    # Ver lo que se esta aprendiendo:
                    image = cv2.imread(personPath+'/'+fileName,0)
                    cv2.imshow('image',image)
                    # eso fue lo que aprendio de
                    cv2.waitKey(3)
                label = label + 1
            
            face_recognizer = cv2.face.LBPHFaceRecognizer_create()
            
            # Entrenando el reconocedor de rostros
            print("Entrenando...")
            face_recognizer.train(facesData, np.array(labels))
            
            # Almacenando el modelo obtenido
            face_recognizer.write('modeloLBPHFace.xml')
            print("¡Modelo almacenado!")
            itemsInfo.append('¡Modelo almacenado!')
            window['-LIST-'].update(itemsInfo)
            cv2.destroyAllWindows()
            #'''
            window['-LIST1-'].update(rostros)
            window.refresh()
        if resultado=="Cancel":
            itemsInfo.append('Entrenamiento Cancelado!!')
            window['-LIST-'].update(itemsInfo)
    
    # 3ro Detección de rostros
    if event == 'Detectar':
        print('Detectando...')
        dataPath = 'imagenes' #Cambia a la ruta donde hayas almacenado Data
        numCamara = 0
        imagePaths = os.listdir(dataPath)
        print('imagePaths=',imagePaths)
        #Reconocimiento de rostro
        face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Leyendo el modelo
        face_recognizer.read('modeloLBPHFace.xml')
        
        #Abriendo Camara
        cap = cv2.VideoCapture(numCamara,cv2.CAP_DSHOW)
        #url = 'http://192.168.100.7:4747/video'
        #cap = cv2.VideoCapture(url)
        #cap = cv2.VideoCapture('Video.mp4')
        
        faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
        abrir = True
        while abrir:
            ret,frame = cap.read()
            if ret == False: break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            auxFrame = gray.copy()
        
            faces = faceClassif.detectMultiScale(gray,1.3,5)
        
            for (x,y,w,h) in faces:
                rostro = auxFrame[y:y+h,x:x+w]
                rostro = cv2.resize(rostro,(150,150),interpolation= cv2.INTER_CUBIC)
                result = face_recognizer.predict(rostro)
        
                cv2.putText(frame,'{}'.format(result),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)
                # LBPHFace
                if result[1] < 70:
                    cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
                    cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
                else:
                    cv2.putText(frame,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
                    cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
                
            cv2.imshow('frame',frame)
            k = cv2.waitKey(1)
            if k == 27:
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    if event == "Salir" or event == sg.WIN_CLOSED:
        break
window.close()