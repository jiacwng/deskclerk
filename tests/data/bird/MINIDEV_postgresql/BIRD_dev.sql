SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;
SET default_tablespace = '';
SET default_table_access_method = heap;

CREATE TABLE public.account (
    account_id bigint DEFAULT '0'::bigint NOT NULL,
    district_id bigint DEFAULT '0'::bigint,
    frequency text,
    date date
);

ALTER TABLE public.account OWNER TO xiaolongli;

CREATE TABLE public.card (
    card_id bigint DEFAULT '0'::bigint NOT NULL,
    disp_id bigint,
    type text,
    issued date
);

ALTER TABLE public.card OWNER TO xiaolongli;

CREATE TABLE public.client (
    client_id bigint NOT NULL,
    gender text,
    birth_date date,
    district_id bigint
);

ALTER TABLE public.client OWNER TO xiaolongli;

CREATE TABLE public.disp (
    disp_id bigint NOT NULL,
    client_id bigint,
    account_id bigint,
    type text
);

ALTER TABLE public.disp OWNER TO xiaolongli;

CREATE TABLE public.district (
    district_id bigint DEFAULT '0'::bigint NOT NULL,
    a2 text,
    a3 text,
    a4 text,
    a5 text,
    a6 text,
    a7 text,
    a8 bigint,
    a9 bigint,
    a10 real,
    a11 bigint,
    a12 real,
    a13 real,
    a14 bigint,
    a15 bigint,
    a16 bigint
);

ALTER TABLE public.district OWNER TO xiaolongli;

CREATE TABLE public.frpm (
    cdscode text,
    "Academic Year" text,
    "County Code" text,
    "District Code" bigint,
    "School Code" text,
    "County Name" text,
    "District Name" text,
    "School Name" text,
    "District Type" text,
    "School Type" text,
    "Educational Option Type" text,
    "NSLP Provision Status" text,
    "Charter School (Y/N)" bigint,
    "Charter School Number" text,
    "Charter Funding Type" text,
    irc bigint,
    "Low Grade" text,
    "High Grade" text,
    "Enrollment (K-12)" real,
    "Free Meal Count (K-12)" real,
    "Percent (%) Eligible Free (K-12)" real,
    "FRPM Count (K-12)" real,
    "Percent (%) Eligible FRPM (K-12)" real,
    "Enrollment (Ages 5-17)" real,
    "Free Meal Count (Ages 5-17)" real,
    "Percent (%) Eligible Free (Ages 5-17)" real,
    "FRPM Count (Ages 5-17)" real,
    "Percent (%) Eligible FRPM (Ages 5-17)" real,
    "2013-14 CALPADS Fall 1 Certification Status" bigint
);

ALTER TABLE public.frpm OWNER TO xiaolongli;

CREATE TABLE public.loan (
    loan_id bigint DEFAULT '0'::bigint NOT NULL,
    account_id bigint,
    date date,
    amount bigint,
    duration bigint,
    payments real,
    status text
);

ALTER TABLE public.loan OWNER TO xiaolongli;

CREATE TABLE public."order" (
    order_id bigint DEFAULT '0'::bigint NOT NULL,
    account_id bigint,
    bank_to text,
    account_to bigint,
    amount real,
    k_symbol text
);

ALTER TABLE public."order" OWNER TO xiaolongli;

CREATE TABLE public.satscores (
    cds text NOT NULL,
    rtype text,
    sname text,
    dname text,
    cname text,
    enroll12 bigint,
    numtsttakr bigint,
    avgscrread bigint,
    avgscrmath bigint,
    avgscrwrite bigint,
    numge1500 bigint
);

ALTER TABLE public.satscores OWNER TO xiaolongli;

CREATE TABLE public.schools (
    cdscode text,
    ncesdist text,
    ncesschool text,
    statustype text,
    county text,
    district text,
    school text,
    street text,
    streetabr text,
    city text,
    zip text,
    state text,
    mailstreet text,
    mailstrabr text,
    mailcity text,
    mailzip text,
    mailstate text,
    phone text,
    ext text,
    website text,
    opendate date,
    closeddate date,
    charter bigint,
    charternum text,
    fundingtype text,
    doc text,
    doctype text,
    soc text,
    soctype text,
    edopscode text,
    edopsname text,
    eilcode text,
    eilname text,
    gsoffered text,
    gsserved text,
    virtual text,
    magnet bigint,
    latitude real,
    longitude real,
    admfname1 text,
    admlname1 text,
    admemail1 text,
    admfname2 text,
    admlname2 text,
    admemail2 text,
    admfname3 text,
    admlname3 text,
    admemail3 text,
    lastupdate date
);

ALTER TABLE public.schools OWNER TO xiaolongli;

CREATE TABLE public.trans (
    trans_id bigint DEFAULT '0'::bigint NOT NULL,
    account_id bigint DEFAULT '0'::bigint,
    date date,
    type text,
    operation text,
    amount bigint,
    balance bigint,
    k_symbol text,
    bank text,
    account bigint
);

ALTER TABLE public.trans OWNER TO xiaolongli;

COPY public.account (account_id, district_id, frequency, date) FROM stdin;
2	1	POPLATEK MESICNE	1993-02-26
71	1	POPLATEK MESICNE	1994-03-05
95	1	POPLATEK MESICNE	1995-03-30
281	1	POPLATEK MESICNE	1996-10-27
300	1	POPLATEK MESICNE	1993-06-23
478	58	POPLATEK MESICNE	1994-07-16
675	58	POPLATEK MESICNE	1995-08-03
2803	58	POPLATEK MESICNE	1995-12-01
3448	58	POPLATEK TYDNE	1996-06-06
3538	58	POPLATEK MESICNE	1993-08-08
\.

COPY public.card (card_id, disp_id, type, issued) FROM stdin;
12	83	junior	1996-09-11
14	112	classic	1996-02-17
56	339	classic	1998-11-22
59	365	classic	1997-10-27
102	575	classic	1998-11-12
142	810	classic	1998-07-01
534	3387	classic	1997-01-29
661	4270	classic	1995-07-03
\.

COPY public.client (client_id, gender, birth_date, district_id) FROM stdin;
2	M	1945-02-04	1
3	F	1940-10-09	1
83	F	1978-12-25	1
84	M	1979-05-05	1
112	M	1935-01-25	1
113	F	1940-08-15	1
339	M	1966-07-30	1
340	F	1970-08-19	1
365	M	1930-01-09	1
366	F	1936-11-10	1
575	M	1970-04-04	58
576	F	1974-05-12	58
810	F	1945-09-24	58
811	M	1940-05-28	58
3387	M	1943-08-14	58
4169	F	1980-02-19	58
4170	M	1983-08-27	58
4270	F	1972-02-05	58
4271	M	1964-07-23	58
\.

COPY public.disp (disp_id, client_id, account_id, type) FROM stdin;
2	2	2	OWNER
3	3	2	DISPONENT
83	83	71	OWNER
84	84	71	DISPONENT
112	112	95	OWNER
113	113	95	DISPONENT
339	339	281	OWNER
340	340	281	DISPONENT
365	365	300	OWNER
366	366	300	DISPONENT
575	575	478	OWNER
576	576	478	DISPONENT
810	810	675	OWNER
811	811	675	DISPONENT
3387	3387	2803	OWNER
4169	4169	3448	OWNER
4170	4170	3448	DISPONENT
4270	4270	3538	OWNER
4271	4271	3538	DISPONENT
\.

COPY public.district (district_id, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16) FROM stdin;
1	Hl.m. Praha	Prague	1204953	0	0	0	1	1	100	12541	0.2	0.43	167	85677	99107
2	Benesov	central Bohemia	88884	80	26	6	2	5	46.7	8507	1.6	1.85	132	2159	2674
3	Beroun	central Bohemia	75232	55	26	4	1	5	41.7	8980	1.9	2.21	111	2824	2813
4	Kladno	central Bohemia	149893	63	29	6	2	6	67.4	9753	4.6	5.05	109	5244	5892
5	Kolin	central Bohemia	95616	65	30	4	1	6	51.4	9307	3.8	4.43	118	2616	3040
6	Kutna Hora	central Bohemia	77963	60	23	4	2	4	51.5	8546	2.9	4.02	126	2640	3120
7	Melnik	central Bohemia	94725	38	28	1	3	6	63.4	9920	2.2	2.87	130	4289	4846
8	Mlada Boleslav	central Bohemia	112065	95	19	7	1	8	69.4	11277	1.2	1.44	127	5179	4987
9	Nymburk	central Bohemia	81344	61	23	4	2	6	55.3	8899	3.3	3.97	149	2987	2487
10	Praha - vychod	central Bohemia	92084	55	29	4	3	5	46.7	10124	0.5	0.54	141	3810	4316
11	Praha - zapad	central Bohemia	75637	35	36	9	0	7	36.5	9622	0.4	0.59	154	3475	3529
12	Pribram	central Bohemia	107870	84	29	6	1	6	58	8754	3.8	4.31	137	3804	3868
13	Rakovnik	central Bohemia	53921	61	22	1	1	2	41.3	8598	2.7	3.26	123	1597	1875
14	Ceske Budejovice	south Bohemia	177686	69	27	10	1	9	74.8	10045	1.4	1.71	135	6604	6295
15	Cesky Krumlov	south Bohemia	58796	22	16	7	1	5	51.9	9045	3.1	3.6	124	1845	1879
16	Jindrichuv Hradec	south Bohemia	93931	74	21	10	1	8	56.9	8427	1.1	1.54	107	1874	1913
17	Pelhrimov	south Bohemia	74062	99	15	4	2	7	61.4	8114	2.3	2.62	119	1003	1181
18	Pisek	south Bohemia	70699	60	13	2	1	4	65.3	8968	2.8	3.35	131	1740	1910
19	Prachatice	south Bohemia	51428	50	11	3	1	4	52.7	8402	3.1	3.98	120	999	1099
20	Strakonice	south Bohemia	70646	94	14	3	1	4	58.4	8547	2.6	3.64	120	1563	1542
21	Tabor	south Bohemia	103347	87	16	7	1	7	67	9104	1.5	2.07	123	2299	2354
22	Domazlice	west Bohemia	58400	65	16	4	1	6	52.4	8620	1.1	1.25	100	1089	1117
23	Cheb	west Bohemia	87419	21	11	4	3	9	85.5	8624	1.7	2.66	102	2879	3198
24	Karlovy Vary	west Bohemia	122603	25	21	6	2	8	80	8991	1.3	2.01	128	5198	5273
25	Klatovy	west Bohemia	88757	60	33	3	2	10	61.7	8554	2.4	2.68	113	1822	2218
26	Plzen - mesto	west Bohemia	170449	0	0	0	1	1	100	10787	2.6	3.09	131	6041	6261
27	Plzen - jih	west Bohemia	67298	71	19	10	0	7	43.8	8561	0.6	1.29	110	1029	1127
28	Plzen - sever	west Bohemia	72541	66	26	10	0	8	36.9	8594	1.6	2.01	112	1580	1525
29	Rokycany	west Bohemia	45714	52	10	5	1	6	55.6	8843	2.8	3.6	113	818	888
30	Sokolov	west Bohemia	94812	15	13	8	2	10	81.8	9650	3.3	3.67	100	2985	2804
31	Tachov	west Bohemia	51313	34	12	3	1	4	59.2	8930	3.5	4.2	108	1328	1452
32	Ceska Lipa	north Bohemia	105058	32	19	7	2	10	81	9272	2.8	3.22	118	4340	4505
33	Decin	north Bohemia	133777	24	17	7	3	11	84.7	8705	5.7	7.61	116	4650	4859
34	Chomutov	north Bohemia	125236	28	11	1	4	5	87.7	9675	6.4	7.68	100	5323	5190
35	Jablonec n. Nisou	north Bohemia	88768	10	18	5	1	7	80.5	8867	1	1.21	130	3384	3620
36	Liberec	north Bohemia	159617	29	19	8	1	9	85.2	9198	3.3	4.28	131	5796	6132
37	Litomerice	north Bohemia	114006	71	26	6	2	9	62.3	9065	4.4	5.39	123	4147	4166
38	Louny	north Bohemia	85852	41	23	4	2	4	59.8	8965	7	8.23	104	2653	2822
39	Most	north Bohemia	119895	17	4	3	2	4	89.9	10446	7.3	9.4	90	4947	4743
40	Teplice	north Bohemia	128118	9	16	6	3	8	85.3	9317	6.4	7.07	97	6949	6872
41	Usti nad Labem	north Bohemia	118650	8	8	5	1	3	85.6	9832	3.3	4.48	114	6445	5471
42	Havlickuv Brod	east Bohemia	95907	87	25	5	2	7	59.1	8388	2.4	2.94	87	1658	1668
43	Hradec Kralove	east Bohemia	161854	73	21	6	1	6	76.3	9425	1.7	2.5	125	4085	3891
44	Chrudim	east Bohemia	105606	77	26	7	2	7	53	8254	2.7	3.76	97	2166	2325
45	Jicin	east Bohemia	77917	85	19	6	1	5	53.5	8390	2.2	2.89	132	2080	2122
46	Nachod	east Bohemia	112709	48	20	7	3	10	73.5	8369	1.7	2.31	117	2854	2618
47	Pardubice	east Bohemia	162580	83	26	5	1	6	72.8	9538	1.5	1.81	111	6079	5410
48	Rychnov nad Kneznou	east Bohemia	78955	50	24	8	1	9	59	9060	1.7	2.44	124	1655	1717
49	Semily	east Bohemia	75685	31	28	5	1	9	60.8	8208	1.8	2.26	155	1660	2123
50	Svitavy	east Bohemia	102609	75	33	2	3	6	52	8187	4.8	5.45	94	2123	2134
51	Trutnov	east Bohemia	121947	37	28	7	3	11	70.5	8541	2.5	2.97	131	3496	3839
52	Usti nad Orlici	east Bohemia	139012	59	41	8	3	10	61.9	8363	2.5	3.49	108	2564	2799
53	Blansko	south Moravia	107911	88	33	6	2	7	50.9	8240	2.5	3.56	99	1850	1903
54	Brno - mesto	south Moravia	387570	0	0	0	1	1	100	9897	1.6	1.96	140	18721	18696
55	Brno - venkov	south Moravia	157042	49	70	18	0	9	33.9	8743	1.8	2.43	111	3659	3894
56	Breclav	south Moravia	124605	11	42	14	1	8	46	8772	4.6	4.98	126	2854	3158
57	Hodonin	south Moravia	161954	21	37	20	3	8	48	8720	3.7	4.5	116	3729	3651
58	Jihlava	south Moravia	109164	98	16	6	1	4	63.6	8757	3.3	3.95	96	2212	2471
59	Kromeriz	south Moravia	108871	41	29	7	2	6	62.1	8444	3.2	3.47	106	2595	2305
60	Prostejov	south Moravia	110643	49	41	4	1	4	51.9	8441	3.4	4.48	115	1879	2252
61	Trebic	south Moravia	117897	139	28	5	1	6	53.8	8814	4.7	5.74	107	2112	2059
62	Uherske Hradiste	south Moravia	145688	22	41	13	2	5	40.7	8544	1.2	1.86	109	2719	2906
63	Vyskov	south Moravia	86513	38	36	5	1	5	50.5	8288	3.7	4.52	110	1562	1460
64	Zlin	south Moravia	197099	29	41	10	2	10	74.7	9624	1.6	2.31	145	4484	4265
65	Znojmo	south Moravia	114200	101	41	4	1	4	43.7	8403	5.7	5.72	105	2157	2718
66	Zdar nad Sazavou	south Moravia	125832	151	41	2	3	5	48.3	8512	3.5	4.12	102	2247	2103
67	Bruntal	north Moravia	106054	38	25	6	2	6	63.1	8110	5.7	6.55	109	3244	3079
68	Frydek - Mistek	north Moravia	228848	15	40	18	2	6	57.2	9893	4	4.72	96	5623	5887
69	Jesenik	north Moravia	42821	4	13	5	1	3	48.4	8173	\N	7.01	124	\N	1358
70	Karvina	north Moravia	285387	0	2	8	5	7	89.9	10177	6.6	7.75	81	9878	10108
71	Novy Jicin	north Moravia	161227	5	35	11	4	10	69.7	8678	5.9	5.57	102	4980	4595
72	Olomouc	north Moravia	226122	32	50	7	4	4	62.6	8994	3.8	4.79	110	9672	9208
73	Opava	north Moravia	182027	17	49	12	2	7	56.4	8746	3.3	3.74	90	4355	4433
74	Ostrava - mesto	north Moravia	323870	0	0	0	1	1	100	10673	4.7	5.44	100	18782	18347
75	Prerov	north Moravia	138032	67	30	4	2	5	64.6	8819	5.3	5.66	99	4063	4505
76	Sumperk	north Moravia	127369	31	32	13	2	7	51.2	8369	4.7	5.88	107	3736	2807
77	Vsetin	north Moravia	148545	8	35	12	3	4	53.5	8909	4	5.56	113	3460	3590
\.

COPY public.frpm (cdscode, "Academic Year", "County Code", "District Code", "School Code", "County Name", "District Name", "School Name", "District Type", "School Type", "Educational Option Type", "NSLP Provision Status", "Charter School (Y/N)", "Charter School Number", "Charter Funding Type", irc, "Low Grade", "High Grade", "Enrollment (K-12)", "Free Meal Count (K-12)", "Percent (%) Eligible Free (K-12)", "FRPM Count (K-12)", "Percent (%) Eligible FRPM (K-12)", "Enrollment (Ages 5-17)", "Free Meal Count (Ages 5-17)", "Percent (%) Eligible Free (Ages 5-17)", "FRPM Count (Ages 5-17)", "Percent (%) Eligible FRPM (Ages 5-17)", "2013-14 CALPADS Fall 1 Certification Status") FROM stdin;
22102232230050	2014-2015	22	10223	2230050	Mariposa	Mariposa County Office of Education	Spring Hill Opportunity	County Office of Education (COE)	Opportunity Schools	Opportunity School	\N	0	\N	\N	0	K	8	13	6	0.46153846	6	0.46153846	13	6	0.46153846	6	0.46153846	1
22102232230100	2014-2015	22	10223	2230100	Mariposa	Mariposa County Office of Education	County Community	County Office of Education (COE)	County Community	County Community School	\N	0	\N	\N	0	K	12	16	13	0.8125	14	0.875	14	13	0.9285714	13	0.9285714	1
22102236105308	2014-2015	22	10223	6105308	Mariposa	Mariposa County Office of Education	Monarch Academy	County Office of Education (COE)	Special Education Schools (Public)	Special Education School	\N	0	\N	\N	0	K	12	33	15	0.45454547	21	0.6363636	27	15	0.5555556	19	0.7037037	1
22655320123307	2014-2015	22	65532	0123307	Mariposa	Mariposa County Unified	Jessie Benton Fremont Community Day	Unified School District	District Community Day Schools	Community Day School	\N	0	\N	\N	0	7	12	2	\N	\N	\N	\N	2	\N	\N	\N	\N	1
22655320125823	2014-2015	22	65532	0125823	Mariposa	Mariposa County Unified	Sierra Foothill Charter	Unified School District	Elementary Schools (Public)	Traditional	\N	1	1396	Directly funded	1	K	8	143	34	0.23776224	54	0.37762237	142	34	0.23943663	54	0.3802817	1
22655322230068	2014-2015	22	65532	2230068	Mariposa	Mariposa County Unified	Sierra Home	Unified School District	Alternative Schools of Choice	Alternative School of Choice	\N	0	\N	\N	0	K	12	47	18	0.38297874	19	0.40425533	43	17	0.39534885	17	0.39534885	1
22655322230076	2014-2015	22	65532	2230076	Mariposa	Mariposa County Unified	Coulterville High	Unified School District	High Schools (Public)	Traditional	\N	0	\N	\N	0	9	12	2	2	1	2	1	1	1	1	1	1	1
22655322230084	2014-2015	22	65532	2230084	Mariposa	Mariposa County Unified	Yosemite Park High	Unified School District	High Schools (Public)	Traditional	\N	0	\N	\N	0	9	12	2	\N	\N	\N	\N	2	\N	\N	\N	\N	1
46701774630034	2014-2015	46	70177	4630034	Sierra	Sierra-Plumas Joint Unified	Sierra Pass (Continuation)	Unified School District	Continuation High Schools	Continuation School	\N	0	\N	\N	0	9	12	5	2	0.4	2	0.4	4	2	0.5	2	0.5	1
46701774632303	2014-2015	46	70177	4632303	Sierra	Sierra-Plumas Joint Unified	Downieville Junior-Senior High	Unified School District	High Schools (Public)	Traditional	\N	0	\N	\N	0	7	12	18	10	0.5555556	11	0.6111111	18	10	0.5555556	11	0.6111111	1
46701774634259	2014-2015	46	70177	4634259	Sierra	Sierra-Plumas Joint Unified	Loyalton High	Unified School District	High Schools (Public)	Traditional	\N	0	\N	\N	0	7	12	143	34	0.23776224	50	0.34965035	139	32	0.23021583	48	0.34532374	1
46701776050611	2014-2015	46	70177	6050611	Sierra	Sierra-Plumas Joint Unified	Downieville Elementary	Unified School District	Elementary Schools (Public)	Traditional	\N	0	\N	\N	0	K	6	30	13	0.43333334	15	0.5	30	13	0.43333334	15	0.5	1
46701776050629	2014-2015	46	70177	6050629	Sierra	Sierra-Plumas Joint Unified	Loyalton Elementary	Unified School District	Elementary Schools (Public)	Traditional	\N	0	\N	\N	0	K	6	176	51	0.28977272	74	0.42045453	176	51	0.28977272	74	0.42045453	1
\.

COPY public.loan (loan_id, account_id, date, amount, duration, payments, status) FROM stdin;
4959	2	1994-01-05	80952	24	3373	A
5110	675	1997-04-01	102240	60	1704	C
5681	3448	1997-10-20	68664	24	2861	C
\.

COPY public."order" (order_id, account_id, bank_to, account_to, amount, k_symbol) FROM stdin;
29402	2	ST	89597016	3372.7	UVER
29403	2	QR	13943797	7266	SIPO
29510	71	YZ	53903484	968	SIPO
29511	71	IJ	11544728	1602	
29512	71	OP	9098768	1011	POJISTNE
29513	71	QR	34794885	778	
29825	281	CD	31156671	5578	SIPO
29854	300	YZ	46045607	1628	SIPO
30124	478	EF	9013551	8137	SIPO
30392	675	MN	74623369	6358	SIPO
30393	675	MN	56463271	1704	UVER
33547	2803	KL	18263268	1904	LEASING
34502	3448	AB	16375589	2860.7	UVER
\.

COPY public.satscores (cds, rtype, sname, dname, cname, enroll12, numtsttakr, avgscrread, avgscrmath, avgscrwrite, numge1500) FROM stdin;
22655322230068	S	Sierra Home	Mariposa County Unified	Mariposa	11	0	\N	\N	\N	\N
46701774632303	S	Downieville Junior-Senior High	Sierra-Plumas Joint Unified	Sierra	4	1	\N	\N	\N	\N
46701774634259	S	Loyalton High	Sierra-Plumas Joint Unified	Sierra	16	9	\N	\N	\N	\N
\.

COPY public.schools (cdscode, ncesdist, ncesschool, statustype, county, district, school, street, streetabr, city, zip, state, mailstreet, mailstrabr, mailcity, mailzip, mailstate, phone, ext, website, opendate, closeddate, charter, charternum, fundingtype, doc, doctype, soc, soctype, edopscode, edopsname, eilcode, eilname, gsoffered, gsserved, virtual, magnet, latitude, longitude, admfname1, admlname1, admemail1, admfname2, admlname2, admemail2, admfname3, admlname3, admemail3, lastupdate) FROM stdin;
22102232230050	0691090	00651	Active	Mariposa	Mariposa County Office of Education	Spring Hill Opportunity	5171 Silva Road	5171 Silva Rd.	Mariposa	95338-9358	CA	5171 Silva Road	5171 Silva Rd.	Mariposa	95338-9358	CA	(209) 742-0290	\N	www.mariposa.k12.ca.us	1980-07-01	\N	0	\N	\N	00	County Office of Education (COE)	13	Opportunity Schools	OPP	Opportunity School	INTMIDJR	Intermediate/Middle/Junior High	K-8	8	N	0	37.472656	-119.902	David	Naranjo	dnaranjo@mcusd.org	\N	\N	\N	\N	\N	\N	2016-06-21
22102232230100	0691090	03303	Active	Mariposa	Mariposa County Office of Education	County Community	5171 Silva Road	5171 Silva Rd.	Mariposa	95338-9358	CA	5171 Silva Road	5171 Silva Rd.	Mariposa	95338-9358	CA	(209) 742-0290	\N	www.mariposa.k12.ca.us	1994-08-29	\N	0	\N	\N	00	County Office of Education (COE)	10	County Community	COMM	County Community School	HS	High School	K-12	10-12	N	0	37.472656	-119.902	David	Naranjo	dnaranjo@mcusd.org	\N	\N	\N	\N	\N	\N	2016-06-21
22102236105308	0691090	08161	Active	Mariposa	Mariposa County Office of Education	Monarch Academy	5082 Old Highway North	5082 Old Highway North	Mariposa	95338	CA	PO Box 8	PO Box 8	Mariposa	95338-0008	CA	(209) 742-0231	\N	www.mariposa.k12.ca.us	1980-07-01	\N	0	\N	\N	00	County Office of Education (COE)	09	Special Education Schools (Public)	SPEC	Special Education School	ELEMHIGH	Elementary-High Combination	K-12	K-12	N	0	37.49027	-119.9645	Marcia	Miller	mlmiller@mcusd.org	\N	\N	\N	\N	\N	\N	2016-06-15
22655320123307	0623940	12770	Active	Mariposa	Mariposa County Unified	Jessie Benton Fremont Community Day	5171 Silva Road	5171 Silva Rd.	Mariposa	95338-9358	CA	PO Box 8	PO Box 8	Mariposa	95338-5001	CA	(209) 742-0290	\N	\N	2010-08-24	\N	0	\N	\N	54	Unified School District	69	District Community Day Schools	COMMDAY	Community Day School	HS	High School	7-12	10	N	0	37.472656	-119.902	David	Naranjo	dnaranjo@mcusd.org	\N	\N	\N	\N	\N	\N	2016-06-21
22655320125823	0623940	13041	Active	Mariposa	Mariposa County Unified	Sierra Foothill Charter	4952 School House Road	4952 School House Rd.	Catheys Valley	95306-9710	CA	4952 School House Road	4952 School House Rd.	Catheys Valley	95306-9710	CA	(209) 742-6222	\N	www.sierrafoothillcharterschool.org	2012-08-22	\N	1	1396	Directly funded	54	Unified School District	60	Elementary Schools (Public)	TRAD	Traditional	ELEM	Elementary	K-8	K-8	N	0	37.42376	-120.10132	Mindy	Bolar	principal@sierrafoothillcharterschool.org	\N	\N	\N	\N	\N	\N	2016-08-22
22655322230068	0623940	06039	Active	Mariposa	Mariposa County Unified	Sierra Home	5171 Silva Road	5171 Silva Rd.	Mariposa	95338-9358	CA	5171 Silva Road	5171 Silva Rd.	Mariposa	95338-9358	CA	(209) 742-0290	\N	www.mariposa.k12.ca.us	1980-07-01	\N	0	\N	\N	54	Unified School District	63	Alternative Schools of Choice	ALTSOC	Alternative School of Choice	ELEMHIGH	Elementary-High Combination	K-12	K-12	N	0	37.472656	-119.902	David	Naranjo	dnaranjo@mcusd.org	\N	\N	\N	\N	\N	\N	2016-06-21
22655322230076	0623940	09576	Active	Mariposa	Mariposa County Unified	Coulterville High	10326 Fiske Road Greeley Hill Route	10326 Fiske Rd. Greeley Hill Rt.	Coulterville	96311-9502	CA	10326 Fiske Road Greeley Hill Route	10326 Fiske Rd. Greeley Hill Rt.	Coulterville	96311-9502	CA	(209) 878-3028	\N	\N	1980-07-01	\N	0	\N	\N	54	Unified School District	66	High Schools (Public)	TRAD	Traditional	HS	High School	9-12	9-12	N	0	37.74624	-120.13071	Tracie	Baughn	tbaughn@mcusd.org	\N	\N	\N	\N	\N	\N	2016-06-15
22655322230084	0623940	09577	Active	Mariposa	Mariposa County Unified	Yosemite Park High	9670 Rancheria Flat Road	9670 Rancheria Flat Rd.	Yosemite	95389-0485	CA	PO Box 49	PO Box 49	El Portal	95318	CA	(209) 372-2414	\N	http://yphs-mcusd-ca.schoolloop.com/	1980-07-01	\N	0	\N	\N	54	Unified School District	66	High Schools (Public)	TRAD	Traditional	HS	High School	9-12	12	N	0	37.672806	-119.80859	Sean	Jacobs	sjacobs@mcusd.org	\N	\N	\N	\N	\N	\N	2016-06-15
46701774630034	0636780	07445	Active	Sierra	Sierra-Plumas Joint Unified	Sierra Pass (Continuation)	700 Fourth Street	700 Fourth St.	Loyalton	96118	CA	PO Box 37	PO Box 37	Loyalton	96118	CA	(530) 993-1660	47	www.sierracountyofficeofeducation.org/	1997-08-25	\N	0	\N	\N	54	Unified School District	68	Continuation High Schools	CON	Continuation School	HS	High School	9-12	12	N	0	39.678905	-120.24503	Thomas	Jones	tjones@spjusd.org	\N	\N	\N	\N	\N	\N	2016-03-11
46701774632303	0636780	06227	Active	Sierra	Sierra-Plumas Joint Unified	Downieville Junior-Senior High	130 School Street	130 School St.	Downieville	95936	CA	PO Box B	PO Box B	Downieville	95936	CA	(530) 289-3473	\N	www.sierracountyofficeofeducation.org/	1980-07-01	\N	0	\N	\N	54	Unified School District	66	High Schools (Public)	TRAD	Traditional	HS	High School	7-12	7-12	P	0	39.561245	-120.82732	Merrill	Grant	mgrant@spjusd.org	\N	\N	\N	\N	\N	\N	2015-06-18
46701774634259	0636780	06229	Active	Sierra	Sierra-Plumas Joint Unified	Loyalton High	700 Fourth Street	700 Fourth St.	Loyalton	96118-0037	CA	PO Box 37	PO Box 37	Loyalton	96118	CA	(530) 993-4454	\N	www.sierracountyofficeofeducation.org/	1980-07-01	\N	0	\N	\N	54	Unified School District	66	High Schools (Public)	TRAD	Traditional	HS	High School	7-12	7-12	P	0	39.678905	-120.24503	Thomas	Jones	tjones@spjusd.org	\N	\N	\N	\N	\N	\N	2015-07-22
46701776050611	0636780	06226	Active	Sierra	Sierra-Plumas Joint Unified	Downieville Elementary	130 School Street	130 School St.	Downieville	95936-0396	CA	PO Box B	PO Box B	Downieville	95936	CA	(530) 289-3473	\N	www.sierracountyofficeofeducation.org/	1980-07-01	\N	0	\N	\N	54	Unified School District	60	Elementary Schools (Public)	TRAD	Traditional	ELEM	Elementary	K-6	K-6	N	0	39.561245	-120.82732	Merrill	Grant	mgrant@spjusd.org	\N	\N	\N	\N	\N	\N	2015-06-18
46701776050629	0636780	06228	Active	Sierra	Sierra-Plumas Joint Unified	Loyalton Elementary	111 Beckwith	111 Beckwith	Loyalton	96118-0127	CA	PO Drawer 127	PO Drawer 127	Loyalton	96118	CA	(530) 993-4482	\N	www.sierracountyofficeofeducation.org/	1980-07-01	\N	0	\N	\N	54	Unified School District	60	Elementary Schools (Public)	TRAD	Traditional	ELEM	Elementary	K-6	K-6	N	0	39.678764	-120.24961	Andrea	White	mgrant@spjusd.org	\N	\N	\N	\N	\N	\N	2016-03-10
\.

COPY public.trans (trans_id, account_id, date, type, operation, amount, balance, k_symbol, bank, account) FROM stdin;
276	2	1993-02-26	PRIJEM	VKLAD	1100	1100	\N	\N	\N
279	2	1993-03-12	PRIJEM	PREVOD Z UCTU	20236	21336	\N	ST	66487163
280	2	1993-04-12	PRIJEM	PREVOD Z UCTU	20236	45286	\N	ST	66487163
281	2	1993-05-12	PRIJEM	PREVOD Z UCTU	20236	54631	\N	ST	66487163
282	2	1993-06-12	PRIJEM	PREVOD Z UCTU	30354	67530	\N	ST	66487163
283	2	1993-07-12	PRIJEM	PREVOD Z UCTU	20236	62326	\N	ST	66487163
284	2	1993-08-12	PRIJEM	PREVOD Z UCTU	20236	49139	\N	ST	66487163
285	2	1993-09-12	PRIJEM	PREVOD Z UCTU	20236	39962	\N	ST	66487163
286	2	1993-10-12	PRIJEM	PREVOD Z UCTU	20236	30744	\N	ST	66487163
287	2	1993-11-12	PRIJEM	PREVOD Z UCTU	20236	40800	\N	ST	66487163
288	2	1993-12-12	PRIJEM	PREVOD Z UCTU	30354	47213	\N	ST	66487163
289	2	1994-01-12	PRIJEM	PREVOD Z UCTU	20236	40811	\N	ST	66487163
290	2	1994-02-12	PRIJEM	PREVOD Z UCTU	20236	33009	\N	ST	66487163
291	2	1994-03-12	PRIJEM	PREVOD Z UCTU	20236	39200	\N	ST	66487163
292	2	1994-04-12	PRIJEM	PREVOD Z UCTU	20236	44012	\N	ST	66487163
293	2	1994-05-12	PRIJEM	PREVOD Z UCTU	20236	46144	\N	ST	66487163
294	2	1994-06-12	PRIJEM	PREVOD Z UCTU	30354	43381	\N	ST	66487163
295	2	1994-07-12	PRIJEM	PREVOD Z UCTU	20236	38690	\N	ST	66487163
296	2	1994-08-12	PRIJEM	PREVOD Z UCTU	20236	43894	\N	ST	66487163
297	2	1994-09-12	PRIJEM	PREVOD Z UCTU	20236	45820	\N	ST	66487163
21305	71	1994-03-05	PRIJEM	VKLAD	700	700	\N	\N	\N
21308	71	1994-03-09	PRIJEM	VKLAD	21717	22417	\N	\N	\N
21309	71	1994-04-09	PRIJEM	VKLAD	21717	44903	\N	\N	\N
21310	71	1994-05-09	PRIJEM	VKLAD	21717	57482	\N	\N	\N
21311	71	1994-06-09	PRIJEM	VKLAD	32576	75084	\N	\N	\N
21312	71	1994-07-09	PRIJEM	VKLAD	21717	71295	\N	\N	\N
21313	71	1994-08-09	PRIJEM	VKLAD	21717	66339	\N	\N	\N
21314	71	1994-09-09	PRIJEM	VKLAD	21717	63950	\N	\N	\N
21315	71	1994-10-09	PRIJEM	VKLAD	21717	59005	\N	\N	\N
21316	71	1994-11-09	PRIJEM	VKLAD	21717	61156	\N	\N	\N
21317	71	1994-12-09	PRIJEM	VKLAD	32576	72887	\N	\N	\N
21318	71	1995-01-09	PRIJEM	VKLAD	21717	52731	\N	\N	\N
21319	71	1995-02-09	PRIJEM	VKLAD	21717	49791	\N	\N	\N
21320	71	1995-03-09	PRIJEM	VKLAD	21717	58014	\N	\N	\N
21321	71	1995-04-09	PRIJEM	VKLAD	21717	60574	\N	\N	\N
21322	71	1995-05-09	PRIJEM	VKLAD	21717	64145	\N	\N	\N
21323	71	1995-06-09	PRIJEM	VKLAD	32576	72882	\N	\N	\N
21324	71	1995-07-09	PRIJEM	VKLAD	21717	67297	\N	\N	\N
21325	71	1995-08-09	PRIJEM	VKLAD	21717	66329	\N	\N	\N
21326	71	1995-09-09	PRIJEM	VKLAD	21717	58411	\N	\N	\N
28865	95	1995-03-30	PRIJEM	VKLAD	200	200	\N	\N	\N
28869	95	1995-04-12	PRIJEM	VKLAD	23927	24127	\N	\N	\N
28870	95	1995-05-12	PRIJEM	VKLAD	23927	48361	\N	\N	\N
28871	95	1995-06-12	PRIJEM	VKLAD	35890	72157	\N	\N	\N
28872	95	1995-07-12	PRIJEM	VKLAD	23927	68739	\N	\N	\N
28873	95	1995-08-12	PRIJEM	VKLAD	23927	67651	\N	\N	\N
28874	95	1995-09-12	PRIJEM	VKLAD	23927	70747	\N	\N	\N
28875	95	1995-10-12	PRIJEM	VKLAD	23927	70554	\N	\N	\N
28876	95	1995-11-12	PRIJEM	VKLAD	23927	66864	\N	\N	\N
28877	95	1995-12-12	PRIJEM	VKLAD	35890	80319	\N	\N	\N
28878	95	1996-01-12	PRIJEM	VKLAD	23927	68262	\N	\N	\N
28879	95	1996-02-12	PRIJEM	VKLAD	23927	59147	\N	\N	\N
28880	95	1996-03-12	PRIJEM	VKLAD	23927	69616	\N	\N	\N
28881	95	1996-04-12	PRIJEM	VKLAD	23927	72322	\N	\N	\N
28882	95	1996-05-12	PRIJEM	VKLAD	23927	74538	\N	\N	\N
28883	95	1996-06-12	PRIJEM	VKLAD	35890	90729	\N	\N	\N
28884	95	1996-07-12	PRIJEM	VKLAD	23927	79012	\N	\N	\N
28885	95	1996-08-12	PRIJEM	VKLAD	23927	83259	\N	\N	\N
28886	95	1996-09-12	PRIJEM	VKLAD	23927	76159	\N	\N	\N
28887	95	1996-10-12	PRIJEM	VKLAD	23927	81327	\N	\N	\N
84243	281	1996-10-27	PRIJEM	VKLAD	900	900	\N	\N	\N
84254	281	1996-11-05	PRIJEM	VKLAD	8367	9267	\N	\N	\N
84255	281	1996-12-05	PRIJEM	VKLAD	12550	31930	\N	\N	\N
84256	281	1997-01-05	PRIJEM	VKLAD	8367	37386	\N	\N	\N
84257	281	1997-02-05	PRIJEM	VKLAD	8367	38074	\N	\N	\N
84258	281	1997-03-05	PRIJEM	VKLAD	8367	44169	\N	\N	\N
84259	281	1997-04-05	PRIJEM	VKLAD	8367	42314	\N	\N	\N
84260	281	1997-05-05	PRIJEM	VKLAD	8367	41253	\N	\N	\N
84261	281	1997-06-05	PRIJEM	VKLAD	12550	50074	\N	\N	\N
84262	281	1997-07-05	PRIJEM	VKLAD	8367	40778	\N	\N	\N
84263	281	1997-08-05	PRIJEM	VKLAD	8367	44777	\N	\N	\N
84264	281	1997-09-05	PRIJEM	VKLAD	8367	54334	\N	\N	\N
84265	281	1997-10-05	PRIJEM	VKLAD	8367	54249	\N	\N	\N
84266	281	1997-11-05	PRIJEM	VKLAD	8367	54814	\N	\N	\N
84267	281	1997-12-05	PRIJEM	VKLAD	12550	62875	\N	\N	\N
84268	281	1998-01-05	PRIJEM	VKLAD	8367	57130	\N	\N	\N
84269	281	1998-02-05	PRIJEM	VKLAD	8367	40425	\N	\N	\N
84270	281	1998-03-05	PRIJEM	VKLAD	8367	52824	\N	\N	\N
84271	281	1998-04-05	PRIJEM	VKLAD	8367	58800	\N	\N	\N
84272	281	1998-05-05	PRIJEM	VKLAD	8367	53759	\N	\N	\N
88982	300	1993-06-23	PRIJEM	VKLAD	400	400	\N	\N	\N
88989	300	1993-07-08	PRIJEM	PREVOD Z UCTU	6892	7292	DUCHOD	YZ	41047549
88990	300	1993-08-08	PRIJEM	PREVOD Z UCTU	6892	14184	DUCHOD	YZ	41047549
88991	300	1993-09-08	PRIJEM	PREVOD Z UCTU	6892	21076	DUCHOD	YZ	41047549
88992	300	1993-10-08	PRIJEM	PREVOD Z UCTU	6892	27968	DUCHOD	YZ	41047549
88993	300	1993-11-08	PRIJEM	PREVOD Z UCTU	6892	33499	DUCHOD	YZ	41047549
88994	300	1993-12-08	PRIJEM	PREVOD Z UCTU	6892	32845	DUCHOD	YZ	41047549
88995	300	1994-01-08	PRIJEM	PREVOD Z UCTU	6892	29688	DUCHOD	YZ	41047549
88996	300	1994-02-08	PRIJEM	PREVOD Z UCTU	6892	24788	DUCHOD	YZ	41047549
88997	300	1994-03-08	PRIJEM	PREVOD Z UCTU	6892	29525	DUCHOD	YZ	41047549
88998	300	1994-04-08	PRIJEM	PREVOD Z UCTU	6892	30259	DUCHOD	YZ	41047549
88999	300	1994-05-08	PRIJEM	PREVOD Z UCTU	6892	28493	DUCHOD	YZ	41047549
89000	300	1994-06-08	PRIJEM	PREVOD Z UCTU	6892	32445	DUCHOD	YZ	41047549
89001	300	1994-07-08	PRIJEM	PREVOD Z UCTU	6892	27308	DUCHOD	YZ	41047549
89002	300	1994-08-08	PRIJEM	PREVOD Z UCTU	6892	29952	DUCHOD	YZ	41047549
89003	300	1994-09-08	PRIJEM	PREVOD Z UCTU	6892	30002	DUCHOD	YZ	41047549
89004	300	1994-10-08	PRIJEM	PREVOD Z UCTU	6892	30752	DUCHOD	YZ	41047549
89005	300	1994-11-08	PRIJEM	PREVOD Z UCTU	6892	28046	DUCHOD	YZ	41047549
89006	300	1994-12-08	PRIJEM	PREVOD Z UCTU	6892	31502	DUCHOD	YZ	41047549
89007	300	1995-01-08	PRIJEM	PREVOD Z UCTU	6892	25428	DUCHOD	YZ	41047549
142732	478	1994-07-16	PRIJEM	VKLAD	400	400	\N	\N	\N
142740	478	1994-08-11	PRIJEM	VKLAD	23664	24064	\N	\N	\N
142741	478	1994-09-11	PRIJEM	VKLAD	23664	47482	\N	\N	\N
142742	478	1994-10-11	PRIJEM	VKLAD	23664	58872	\N	\N	\N
142743	478	1994-11-11	PRIJEM	VKLAD	23664	63894	\N	\N	\N
142744	478	1994-12-11	PRIJEM	VKLAD	35496	80469	\N	\N	\N
142745	478	1995-01-11	PRIJEM	VKLAD	23664	65245	\N	\N	\N
142746	478	1995-02-11	PRIJEM	VKLAD	23664	49144	\N	\N	\N
142747	478	1995-03-11	PRIJEM	VKLAD	23664	48671	\N	\N	\N
142748	478	1995-04-11	PRIJEM	VKLAD	23664	50189	\N	\N	\N
142749	478	1995-05-11	PRIJEM	VKLAD	23664	49777	\N	\N	\N
142750	478	1995-06-11	PRIJEM	VKLAD	35496	56011	\N	\N	\N
142751	478	1995-07-11	PRIJEM	VKLAD	23664	56183	\N	\N	\N
142752	478	1995-08-11	PRIJEM	VKLAD	23664	53167	\N	\N	\N
142753	478	1995-09-11	PRIJEM	VKLAD	23664	51996	\N	\N	\N
142754	478	1995-10-11	PRIJEM	VKLAD	23664	58715	\N	\N	\N
142755	478	1995-11-11	PRIJEM	VKLAD	23664	59958	\N	\N	\N
142756	478	1995-12-11	PRIJEM	VKLAD	35496	74841	\N	\N	\N
142757	478	1996-01-11	PRIJEM	VKLAD	23664	63735	\N	\N	\N
142758	478	1996-02-11	PRIJEM	VKLAD	23664	58208	\N	\N	\N
198803	675	1995-08-03	PRIJEM	VKLAD	900	900	\N	\N	\N
198811	675	1995-08-14	PRIJEM	VKLAD	9538	10438	\N	\N	\N
198812	675	1995-09-14	PRIJEM	VKLAD	9538	26401	\N	\N	\N
198813	675	1995-10-14	PRIJEM	VKLAD	9538	36120	\N	\N	\N
198814	675	1995-11-14	PRIJEM	VKLAD	9538	38692	\N	\N	\N
198815	675	1995-12-14	PRIJEM	VKLAD	14307	43456	\N	\N	\N
198816	675	1996-01-14	PRIJEM	VKLAD	9538	20993	\N	\N	\N
198817	675	1996-02-14	PRIJEM	VKLAD	9538	23386	\N	\N	\N
198818	675	1996-03-14	PRIJEM	VKLAD	9538	29941	\N	\N	\N
198819	675	1996-04-14	PRIJEM	VKLAD	9538	33326	\N	\N	\N
198820	675	1996-05-14	PRIJEM	VKLAD	9538	36527	\N	\N	\N
198821	675	1996-06-14	PRIJEM	VKLAD	14307	33782	\N	\N	\N
198822	675	1996-07-14	PRIJEM	VKLAD	9538	36574	\N	\N	\N
198823	675	1996-08-14	PRIJEM	VKLAD	9538	40585	\N	\N	\N
198824	675	1996-09-14	PRIJEM	VKLAD	9538	31629	\N	\N	\N
198825	675	1996-10-14	PRIJEM	VKLAD	9538	30603	\N	\N	\N
198826	675	1996-11-14	PRIJEM	VKLAD	9538	38094	\N	\N	\N
198827	675	1996-12-14	PRIJEM	VKLAD	14307	40620	\N	\N	\N
198828	675	1997-01-14	PRIJEM	VKLAD	9538	29721	\N	\N	\N
198829	675	1997-02-14	PRIJEM	VKLAD	9538	30246	\N	\N	\N
822925	2803	1995-12-01	PRIJEM	VKLAD	300	300	\N	\N	\N
822933	2803	1995-12-20	PRIJEM	VKLAD	44028	44328	\N	\N	\N
822947	2803	1996-05-08	PRIJEM	VKLAD	32831	93266	\N	\N	\N
822948	2803	1996-12-08	PRIJEM	VKLAD	14075	91321	\N	\N	\N
822949	2803	1996-03-02	PRIJEM	VKLAD	29892	75452	\N	\N	\N
822950	2803	1996-03-15	PRIJEM	VKLAD	11374	79374	\N	\N	\N
822951	2803	1996-10-14	PRIJEM	VKLAD	39359	104528	\N	\N	\N
822952	2803	1996-01-01	PRIJEM	VKLAD	40527	74825	\N	\N	\N
822953	2803	1996-02-12	PRIJEM	VKLAD	22267	102131	\N	\N	\N
822954	2803	1996-02-26	PRIJEM	VKLAD	26467	125875	\N	\N	\N
822955	2803	1996-02-19	PRIJEM	VKLAD	28639	130770	\N	\N	\N
822956	2803	1996-02-01	PRIJEM	VKLAD	20030	66954	\N	\N	\N
822957	2803	1996-04-12	PRIJEM	VKLAD	26629	73660	\N	\N	\N
822958	2803	1996-05-03	PRIJEM	VKLAD	10222	64465	\N	\N	\N
822959	2803	1996-02-01	PRIJEM	VKLAD	12910	79864	\N	\N	\N
822960	2803	1996-05-06	PRIJEM	VKLAD	17733	82198	\N	\N	\N
822961	2803	1996-01-27	PRIJEM	VKLAD	44318	119143	\N	\N	\N
822962	2803	1996-06-12	PRIJEM	VKLAD	19703	75696	\N	\N	\N
822963	2803	1996-11-23	PRIJEM	VKLAD	10739	71584	\N	\N	\N
822964	2803	1996-02-23	PRIJEM	VKLAD	20738	99408	\N	\N	\N
1009426	3448	1996-06-06	PRIJEM	VKLAD	700	700	\N	\N	\N
1009427	3448	1996-06-25	PRIJEM	VKLAD	11546	12246	\N	\N	\N
1009430	3448	1996-09-06	PRIJEM	VKLAD	26017	61268	\N	\N	\N
1009432	3448	1996-11-18	PRIJEM	VKLAD	7310	46369	\N	\N	\N
1009433	3448	1996-10-31	PRIJEM	VKLAD	24509	62589	\N	\N	\N
1009434	3448	1996-07-07	PRIJEM	VKLAD	46159	72394	\N	\N	\N
1009436	3448	1996-07-04	PRIJEM	VKLAD	17079	29335	\N	\N	\N
1009437	3448	1997-06-29	PRIJEM	VKLAD	48806	68478	\N	\N	\N
1009438	3448	1997-10-17	PRIJEM	VKLAD	17809	36642	\N	\N	\N
1009439	3448	1997-08-26	PRIJEM	VKLAD	40614	68423	\N	\N	\N
1009440	3448	1997-01-20	PRIJEM	VKLAD	14412	37785	\N	\N	\N
1009441	3448	1997-04-07	PRIJEM	VKLAD	13057	52861	\N	\N	\N
1009442	3448	1997-04-03	PRIJEM	VKLAD	18492	39804	\N	\N	\N
1009443	3448	1998-03-06	PRIJEM	VKLAD	18929	36101	\N	\N	\N
1009444	3448	1998-03-12	PRIJEM	VKLAD	43252	79353	\N	\N	\N
1009445	3448	1998-04-06	PRIJEM	VKLAD	27413	111583	\N	\N	\N
1009446	3448	1998-06-22	PRIJEM	VKLAD	44664	80944	\N	\N	\N
1009447	3448	1998-03-31	PRIJEM	VKLAD	40584	83976	\N	\N	\N
1009448	3448	1998-07-13	PRIJEM	VKLAD	32356	74398	\N	\N	\N
1009449	3448	1998-10-07	PRIJEM	VKLAD	12192	46930	\N	\N	\N
1034647	3538	1993-08-08	PRIJEM	VKLAD	400	400	\N	\N	\N
1034656	3538	1993-09-05	PRIJEM	VKLAD	23528	23928	\N	\N	\N
1034657	3538	1993-10-05	PRIJEM	VKLAD	23528	47937	\N	\N	\N
1034658	3538	1993-11-05	PRIJEM	VKLAD	23528	61713	\N	\N	\N
1034659	3538	1993-12-05	PRIJEM	VKLAD	35292	77377	\N	\N	\N
1034660	3538	1994-01-05	PRIJEM	VKLAD	23528	75012	\N	\N	\N
1034661	3538	1994-02-05	PRIJEM	VKLAD	23528	52396	\N	\N	\N
1034662	3538	1994-03-05	PRIJEM	VKLAD	23528	76114	\N	\N	\N
1034663	3538	1994-04-05	PRIJEM	VKLAD	23528	72013	\N	\N	\N
1034664	3538	1994-05-05	PRIJEM	VKLAD	23528	68777	\N	\N	\N
1034665	3538	1994-06-05	PRIJEM	VKLAD	35292	72825	\N	\N	\N
1034666	3538	1994-07-05	PRIJEM	VKLAD	23528	52899	\N	\N	\N
1034667	3538	1994-08-05	PRIJEM	VKLAD	23528	59721	\N	\N	\N
1034668	3538	1994-09-05	PRIJEM	VKLAD	23528	66095	\N	\N	\N
1034669	3538	1994-10-05	PRIJEM	VKLAD	23528	69980	\N	\N	\N
1034670	3538	1994-11-05	PRIJEM	VKLAD	23528	71882	\N	\N	\N
1034671	3538	1994-12-05	PRIJEM	VKLAD	35292	85558	\N	\N	\N
1034672	3538	1995-01-05	PRIJEM	VKLAD	23528	77115	\N	\N	\N
1034673	3538	1995-02-05	PRIJEM	VKLAD	23528	60193	\N	\N	\N
1034674	3538	1995-03-05	PRIJEM	VKLAD	23528	68062	\N	\N	\N
\.

ALTER TABLE ONLY public.account
    ADD CONSTRAINT idx_17818_account_pkey PRIMARY KEY (account_id);

ALTER TABLE ONLY public.card
    ADD CONSTRAINT idx_17825_card_pkey PRIMARY KEY (card_id);

ALTER TABLE ONLY public.client
    ADD CONSTRAINT idx_17831_client_pkey PRIMARY KEY (client_id);

ALTER TABLE ONLY public.disp
    ADD CONSTRAINT idx_17836_disp_pkey PRIMARY KEY (disp_id);

ALTER TABLE ONLY public.district
    ADD CONSTRAINT idx_17841_district_pkey PRIMARY KEY (district_id);

ALTER TABLE ONLY public.loan
    ADD CONSTRAINT idx_17847_loan_pkey PRIMARY KEY (loan_id);

ALTER TABLE ONLY public."order"
    ADD CONSTRAINT idx_17853_order_pkey PRIMARY KEY (order_id);

ALTER TABLE ONLY public.trans
    ADD CONSTRAINT idx_17859_trans_pkey PRIMARY KEY (trans_id);

ALTER TABLE ONLY public.satscores
    ADD CONSTRAINT idx_48501_sqlite_autoindex_satscores_1 PRIMARY KEY (cds);

CREATE UNIQUE INDEX idx_48496_sqlite_autoindex_frpm_1 ON public.frpm USING btree (cdscode);

CREATE UNIQUE INDEX idx_48506_sqlite_autoindex_schools_1 ON public.schools USING btree (cdscode);

ALTER TABLE ONLY public.account
    ADD CONSTRAINT account_district_id_fkey FOREIGN KEY (district_id) REFERENCES public.district(district_id);

ALTER TABLE ONLY public.card
    ADD CONSTRAINT card_disp_id_fkey FOREIGN KEY (disp_id) REFERENCES public.disp(disp_id);

ALTER TABLE ONLY public.client
    ADD CONSTRAINT client_district_id_fkey FOREIGN KEY (district_id) REFERENCES public.district(district_id);

ALTER TABLE ONLY public.disp
    ADD CONSTRAINT disp_account_id_fkey FOREIGN KEY (account_id) REFERENCES public.account(account_id);

ALTER TABLE ONLY public.disp
    ADD CONSTRAINT disp_client_id_fkey FOREIGN KEY (client_id) REFERENCES public.client(client_id);

ALTER TABLE ONLY public.frpm
    ADD CONSTRAINT frpm_cdscode_fkey FOREIGN KEY (cdscode) REFERENCES public.schools(cdscode);

ALTER TABLE ONLY public.loan
    ADD CONSTRAINT loan_account_id_fkey FOREIGN KEY (account_id) REFERENCES public.account(account_id);

ALTER TABLE ONLY public."order"
    ADD CONSTRAINT order_account_id_fkey FOREIGN KEY (account_id) REFERENCES public.account(account_id);

ALTER TABLE ONLY public.satscores
    ADD CONSTRAINT satscores_cds_fkey FOREIGN KEY (cds) REFERENCES public.schools(cdscode);

ALTER TABLE ONLY public.trans
    ADD CONSTRAINT trans_account_id_fkey FOREIGN KEY (account_id) REFERENCES public.account(account_id);

