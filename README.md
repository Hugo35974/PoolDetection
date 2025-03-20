# Détection de Personnes autour d'une Piscine

![v1.0](https://img.shields.io/badge/Version-1.0-%2300FF00)

## Description
Ce projet a pour objectif de détecter les personnes adultes et enfants autour d'une piscine à partir d'une vidéo.

## Installation

### Prérequis

- Python 3.x
- OpenCV
- NumPy
- PyTorch 

### Installation des dépendances

Clonez ce repository et installez les dépendances :

```bash
git clone https://github.com/Hugo35974/PoolDetection.git
cd PoolDetection
pip install -r requirements.txt
```
###Mise en place du site par docker-compose

Dans le terminal faire les commandes suivantes :
- docker-compose up -d
- docker exec -it python_app bash

Ensuite aller sur un navigateur de recherche et écrire http://127.0.0.1:8000/
