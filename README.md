<h4>Реализованная функциональность</h4>
<ul>
    <li></li>
</ul>

<h4>Особенность проекта в следующем:</h4>
<ul>
	<li></li>
</ul>
<h4>Основной стек технологий:</h4>
<ul>
    <li>Python 3+.</li>
	<li>HTML, CSS, JavaScript, TypeScript.</li>
	<li>SCSS.</li>
	<li>Gulp, Webpack, Babel.</li>
	<li>Vue.</li>
	<li>Git.</li>
	<li>Heroku.</li>

 </ul>

<h4>Демо</h4>
<p>Демо сервиса доступно по адресу: *url* </p>

Установка и запуск
------------
 Сервер
------
Выполните:

```bash
sudo apt-get update
sudo apt-get install -y software-properties-common python3.9 python3-pip

git clone git@github.com:Nikita-Sherstnev/*repo*.git
cd *repo*/server
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
docker run -d -p 6379:6379 --name realtime-redis redis 
python -m server
```

Endpoints:

/endpoint (GET) - ...

***

 Клиент
------

### Среда запуска

- NodeJS, NPM 14.17.0+ 
- Yarn 1.22.5+

### Установка
```
git clone git@github.com:Nikita-Sherstnev/*repo*.git
cd *repo*/client
yarn install
```

### Компиляция
```
yarn serve
```

### Сборка
```
yarn build
```

## Разработчики

<h4>Шерстнев Никита - Data Science, Backend https://t.me/iewkw</h4>
<h4>Давидович Артем - Full-Stack https://t.me/artyom_d </h4>
<h4>Сандуляк Степан - Backend https://t.me/developmc </h4>