TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
    para.cpp \
    smo.cpp \
    main.cpp

include(deployment.pri)
qtcAddDeployment()

HEADERS += \
    para.h \
    smo.h

