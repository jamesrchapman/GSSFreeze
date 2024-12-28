import json
import pandas as pd
import numpy as np
import os
import ast
from statsmodels.stats.multitest import multipletests

from statsmodels.miscmodels.ordinal_model import OrderedModel
from scipy.stats import chi2_contingency, kruskal
from scipy.stats import spearmanr
import seaborn as sns
import matplotlib.pyplot as plt


gss_csv_file = 'GSS_cumulative_data.csv'

variables=['year', 'id_', 'hrs1', 'hrs2', 'wrkslf', 'occ10', 'sphrs1', 'sphrs2', 'happy', 'hapmar', 'joblose', 'satjob', 'class_', 'satfin', 'finalter', 'tvhours', 'wrktype', 'yearsjob', 'waypaid', 'wrksched', 'moredays', 'mustwork', 'wrkhome', 'whywkhme', 'famwkoff', 'wkvsfam', 'famvswk', 'hrsrelax', 'secondwk', 'learnnew', 'workfast', 'overwork', 'respect', 'trustman', 'proudemp', 'supcares', 'condemnd', 'promtefr', 'cowrkint', 'jobsecok', 'manvsemp', 'trynewjb', 'health1', 'mntlhlth', 'spvtrfair', 'slpprblm', 'satjob1', 'hyperten', 'stress', 'realinc', 'ballot', 'sei10']
# predictor=['hrs1', 'hrs2', 'wrkslf', 'occ10', 'sphrs1', 'sphrs2', 'hapmar', 'joblose', 'satjob', 'class_', 'satfin', 'finalter', 'tvhours', 'wrktype', 'yearsjob', 'waypaid', 'wrksched', 'moredays', 'mustwork', 'wrkhome', 'whywkhme', 'famwkoff', 'wkvsfam', 'famvswk', 'hrsrelax', 'secondwk', 'learnnew', 'workfast', 'overwork', 'respect', 'trustman', 'proudemp', 'supcares', 'condemnd', 'promtefr', 'cowrkint', 'jobsecok', 'manvsemp', 'trynewjb', 'health1', 'mntlhlth', 'spvtrfair', 'slpprblm', 'satjob1', 'hyperten', 'stress', 'realinc', 'ballot', 'sei10']
dependent=['happy', 'health1', 'mntlhlth', 'slpprblm', 'hyperten', 'stress']

def main():
        
    # Load the CSV file into a pandas DataFrame
    try:
        print("Attempting to load the CSV file...")
        df = pd.read_csv(gss_csv_file)
        print("Successfully loaded GSS cumulative data.")
        # print("DataFrame preview:")
        # print(df.head())
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        df = None


    # Analyze intersectional variables if DataFrame is available
    if df is not None:
        try:
            # Select health-related variables for amalgam
            health_vars = ['happy', 'health1', 'mntlhlth', 'slpprblm', 'hyperten', 'stress']
            df_health = df[health_vars]#.replace({
            #     r'\.i:.*': np.nan,
            #     r'\.n:.*': np.nan,
            #     r'\.d:.*': np.nan,
            #     r'\.s:.*': np.nan,
            #     r'\.y:.*': np.nan
            # }, regex=True)

            # # Explore mntlhlth data types and values
            # print("mntlhlth data types and values:")
            # print(df_health['mntlhlth'].apply(lambda x: (x, type(x))))
            # # Manual categorical mapping for each health variable
            custom_mappings = {
                'happy': {'Not too happy': 0, 'Pretty happy': 1, 'Very happy': 2, '.n:  No answer':-100,'.d:  Do not Know/Cannot Choose':-100,'.i:  Inapplicable':-100,'.s:  Skipped on Web':-100,'.y:  Not available in this year':-100},
                'health1': {'Poor': 0, 'Fair': 1, 'Good': 2, 'Very good': 3, 'Excellent': 4,'.n:  No answer':-100,'.d:  Do not Know/Cannot Choose':-100,'.i:  Inapplicable':-100,'.s:  Skipped on Web':-100,'.y:  Not available in this year':-100},
                'mntlhlth': lambda x: 0, #int(x) if isinstance(x, str) and x.isdigit() else np.nan,
                'slpprblm': {'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Often': 3,'.n:  No answer':-100,'.d:  Do not Know/Cannot Choose':-100,'.i:  Inapplicable':-100,'.s:  Skipped on Web':-100,'.y:  Not available in this year':-100},
                'hyperten': {'No': 0, 'Yes': 1,'.n:  No answer':-100,'.d:  Do not Know/Cannot Choose':-100,'.i:  Inapplicable':-100,'.s:  Skipped on Web':-100,'.y:  Not available in this year':-100},
                'stress': {'Never': 0, 'Hardly ever': 1, 'Sometimes': 2, 'Often': 3, 'Always': 4,'.n:  No answer':-100,'.d:  Do not Know/Cannot Choose':-100,'.i:  Inapplicable':-100,'.s:  Skipped on Web':-100,'.y:  Not available in this year':-100}
            }
            
            # Apply mappings
            for var, mapping in custom_mappings.items():
                if callable(mapping):
                    df_health[var] = df_health[var].apply(mapping)
                else:
                    df_health[var] = df_health[var].map(lambda x:-100 if x == np.nan else mapping[x])
                    print(f"Custom categorical mapping for {var}: {mapping}")
            
            df_health['health_score'] = df_health.sum(axis=1)
            print(df_health[['health_score']].describe())

            # Print out examples of how health score is calculated
            print("Examples of health score calculation:")
            print(df_health[health_vars + ['health_score']].head(10))

            # Filter for predictor variables
            predictors = [col for col in df.columns if col not in health_vars]
            df_predictors = df[predictors].dropna()

            # Ordinal Logistic Regression to predict health amalgam
            model = OrderedModel(df_health['health_score'], df_predictors, distr='logit')
            result = model.fit(method='bfgs')
            print(result.summary())
            
            # AIC/BIC evaluation
            print(f"Health Model - AIC: {result.aic}, BIC: {result.bic}")

            # print("Unique responses for each variable:")
            # for column in df.columns:
            #     unique_values = df[column].dropna().unique()
            #     print(f"{column}: {unique_values}")
            # # print("Selecting key variables for analysis...")
            # # Pick 3 key variables to examine based on potential insights into freeze and stress

            # # Define the correct order for each ordinal variable
            # # stress_order = ["Never", "Hardly ever", "Sometimes", "Often", "Always"]
            # # joblose_order = ["Not likely", "Not too likely", "Fairly likely", "Very likely"]
            # # satfin_order = ["Not satisfied at all", "More or less satisfied", "Pretty well satisfied"]

            # # Replace inapplicable and non-answer values with NaN for analysis
            # df_subset = df[dependent].replace({
            #     r'\.i:.*': np.nan, 
            #     r'\.n:.*': np.nan, 
            #     r'\.d:.*': np.nan, 
            #     r'\.s:.*': np.nan,
            #     r'\.y:.*': np.nan
            # }, regex=True)

            # # Convert ordinal columns to categorical types with the defined order
            # # df_subset['stress'] = pd.Categorical(df_subset['stress'], categories=stress_order, ordered=True)
            # # df_subset['joblose'] = pd.Categorical(df_subset['joblose'], categories=joblose_order, ordered=True)
            # # df_subset['satfin'] = pd.Categorical(df_subset['satfin'], categories=satfin_order, ordered=True)

            # # Drop rows with any NaN values to only keep rows with complete data
            # df_filtered = df_subset.dropna()


            # # Display the filtered data
            # print("Filtered DataFrame preview:")
            # print(df_filtered.head())

            # Convert categorical data to numeric rankings for correlation analysis
            # df_numeric = df_filtered.apply(lambda x: x.cat.codes)
        except KeyError as e:
            print(f"Error: Some variables are not found in the DataFrame: {e}")
        except Exception as e:
            print(f"Error analyzing selected variables: {e}")

if __name__ == '__main__':
    main()

    """
I'm looking at a bunch of variables from a survey and how they're going to predict these dependent variables which are also survey questions. 
So, the issue I'm running into is that the survey questions are different formats right - like what I want to do is find all of the variables that are convincingly associated with variation in the dependent variables (independent of each other). 
So I need help figuring out how to find significance between different types of survey questions and the predictor variables - e.g. how do I associate nominal independent with ordinal dependent. How do I associate ordinal with nominal, ratio with ordinal, all combinations. 

Then I want to hold onto those variables for each predictor. 
Then the real kicker is looking to associate each validated predictor variable with the dependent variables in all pairs over those variables. I want to look at some kind of regression model and to examine the bayes information criterion or AIC or whatever. Something like that, to see if the interaction of the pairs explains a good deal more of the variance than independently. 
I don't know what kind of regression to do for the different question formats again! how do you predict ordinal variables with a pair of nominal and ordinal inputs? etc. 

and then there's a validation issue where I should probably look at random data to make sure it's not just an artifact of the model. 




Regression by Dependent Variable Type
Ordinal Dependent Variable: Use Ordinal Logistic Regression. This can handle both nominal and ratio predictors, as well as interaction terms.
Nominal Dependent Variable: Use Multinomial Logistic Regression.
Ratio Dependent Variable: Use Linear Regression or Generalized Linear Models (GLMs) for non-normal distributions.

Including Interaction Terms
To evaluate interactions:

Create interaction terms by multiplying predictors
Use a regression model suited to your dependent variable (e.g., ordinal logistic regression for ordinal outcomes).
Compare model fits using Akaike Information Criterion (AIC) or Bayesian Information Criterion (BIC). Models with lower AIC/BIC are preferred, but ensure they’re not overfitting.







Unique responses for each variable:
year: [1972 1973 1974 1975 1976 1977 1978 1980 1982 1983 1984 1985 1986 1987
 1988 1989 1990 1991 1993 1994 1996 1998 2000 2002 2004 2006 2008 2010
 2012 2014 2016 2018 2021 2022]
id_: [   1    2    3 ... 4508 4509 4510]
hrs1: ['.i:  Inapplicable' '27' '40' '52' '35' '45' '70' '60' '42' '23' '55'
 '50' '6' '37' '36' '.d:  Do not Know/Cannot Choose' '48' '16' '24' '20'
 '80' '65' '15' '39' '25' '22' '44' '26' '38' '12' '41' '84' '18' '54'
 '32' '3' '29' '51' '30' '.n:  No answer' '89+ hrs' '43' '58' '28' '46'
 '8' '9' '19' '14' '66' '61' '49' '10' '72' '34' '75' '53' '7' '21' '47'
 '57' '71' '13' '5' '31' '17' '56' '4' '0' '33' '1' '68' '11' '2' '85'
 '62' '59' '78' '77' '64' '63' '67' '73' '88' '69' '76' '74' '81' '79'
 '86' '87' '83' '82' '.s:  Skipped on Web']
hrs2: ['.i:  Inapplicable' '40' '55' '15' '56' '50' '10' '35' '.n:  No answer'
 '70' '16' '37' '30' '48' '20' '5' '25' '60' '32' '52' '45'
 '.d:  Do not Know/Cannot Choose' '75' '24' '4' '89+ hrs' '44' '34' '13'
 '8' '12' '6' '80' '39' '21' '47' '36' '38' '33' '46' '42' '43' '11' '66'
 '23' '7' '58' '18' '65' '84' '17' '68' '41' '1' '2' '0' '72' '28' '57'
 '3' '22' '27' '26' '9' '.s:  Skipped on Web']
wrkslf: ['Someone else' '.i:  Inapplicable' 'Self-employed' '.n:  No answer'
 '.d:  Do not Know/Cannot Choose' '.s:  Skipped on Web']
occ10: ['Wholesale and retail buyers, except farm products'
 'First-line supervisors of production and operating workers'
 'Real estate brokers and sales agents' 'Accountants and auditors'
 'Telephone operators'
 'Sales representatives,  wholesale and manufacturing'
 'Pipelayers, plumbers, pipefitters, and steamfitters' 'Order clerks'
 'Cooks' 'Maids and housekeeping cleaners'
 'Pressers, textile, garment, and related materials'
 'Baggage porters, bellhops, and concierges' 'Couriers and messengers'
 'Tire builders' 'Court, municipal, and license clerks'
 'Secondary school teachers' '.i:  Inapplicable' 'Construction managers'
 'Locomotive engineers and operators'
 'Driver/sales workers and truck drivers'
 'Secretaries and administrative assistants'
 'Elementary and middle school teachers' 'Packers and packagers, hand'
 'Retail salespersons'
 'Inspectors, testers, sorters, samplers, and weighers'
 'Automotive and watercraft service attendants'
 'Welding, soldering, and brazing workers'
 'Plating and coating machine setters, operators, and tenders, metal  and plastic'
 'Registered nurses' 'Military, rank not specified'
 'Paper goods machine setters, operators, and tenders'
 'Farmers, ranchers, and other agricultural managers'
 'Postsecondary teachers'
 'Crushing, grinding, polishing, mixing, and blending workers'
 'Hairdressers, hairstylists, and cosmetologists'
 'Public relations specialists' 'Carpenters' 'Bakers'
 'Carpet, floor, and tile installers and finishers'
 'Printing press operators'
 'Industrial and refractory machinery mechanics' 'Tellers'
 'Miscellaneous assemblers and fabricators' 'Office clerks, general'
 'Cutting workers' 'Miscellaneous agricultural workers' '.u:  Uncodable'
 'Production workers, all other' 'Sewing machine operators'
 'First-line supervisors of mechanics, installers, and repairers'
 'Bookkeeping, accounting, and auditing clerks'
 'Word processors and typists' 'Office machine operators, except computer'
 'Waiters and waitresses' 'Maintenance workers, machinery'
 'Drywall installers, ceiling tile installers, and tapers'
 'Financial analysts' 'Chemical engineers' 'Human resources workers'
 'Nursing, psychiatric, and home health aides'
 'Military enlisted tactical operations and air/weapons specialists and crew members'
 'Financial managers' 'Grounds maintenance workers'
 'First-line supervisors of retail sales workers' 'Painting workers'
 'Grinding, lapping, polishing, and buffing machine tool setters,  operators, and tenders, metal and plastic'
 'Construction laborers' 'Artists and related workers'
 'Hotel, motel, and resort desk clerks' 'Clergy'
 'Janitors and building cleaners'
 'Cutting, punching, and press machine setters, operators, and tenders, metal and plastic'
 'Drafters' 'Electrical and electronics repairers, industrial and utility'
 'Electricians' 'Stationary engineers and boiler operators'
 'Chemical processing machine setters, operators, and tenders'
 'Communications  equipment operators, all other'
 "Police and sheriff's patrol officers"
 'Operating engineers and other construction equipment  operators'
 'Office and administrative support workers, all other'
 'Electric motor, power tool, and related repairers'
 'Computer systems analysts' 'Chief executives' 'Cashiers'
 'Print binding and finishing workers'
 'Appraisers and assessors of real estate'
 'First-line supervisors of food preparation and serving workers'
 'Laborers and freight, stock, and material movers, hand'
 'Other teachers and instructors' 'Purchasing managers'
 'Producers and directors' 'Dental hygienists' 'Education administrators'
 'First-line supervisors of office and administrative support workers'
 'Social and human service assistants' 'Customer service representatives'
 'Machinists' 'Postal service mail carriers'
 'Laundry and dry-cleaning workers' 'Highway maintenance workers'
 'Childcare workers' 'Marketing and sales managers'
 'Industrial production managers' 'Motion picture projectionists'
 'Electrical, electronics, and electromechanical assemblers'
 'Shoe machine operators and tenders' 'Dentists' 'Lodging managers'
 'Gaming cage workers' 'Sheet metal workers' 'Postal service clerks'
 'Managers, all other' 'Firefighters'
 'Butchers and other meat, poultry, and fish processing workers'
 'Weighers, measurers, checkers, and samplers, recordkeeping'
 'Metal workers and plastic workers, all other'
 'Shipping, receiving, and traffic clerks' 'Crane and tower operators'
 'Tool and die makers' 'Writers and authors'
 'Surveying and mapping technicians'
 'Compensation, benefits, and job analysis specialists'
 'Structural iron and steel workers'
 'Hosts and hostesses, restaurant, lounge, and coffee shop'
 'Electrical and electronics engineers'
 'Elevator installers and repairers' 'Food preparation workers'
 'Cleaners of vehicles and equipment'
 'Refuse and recyclable material collectors'
 'Aircraft structure, surfaces, rigging, and systems assemblers'
 'Painters, construction and maintenance' 'Insurance sales agents'
 'Social and community service managers'
 'Industrial truck and tractor operators'
 'Textile winding, twisting, and drawing out machine setters, operators,  and tenders'
 'First-line supervisors of construction trades and extraction workers'
 'Security guards and gaming surveillance officers'
 'Lathe and turning machine tool setters, operators, and tenders, metal and plastic'
 'Chemical technicians' 'Stock clerks and order fillers'
 'Preschool and kindergarten teachers' 'Food service managers'
 'Bus drivers' 'Helpers, construction trades'
 'Cement masons, concrete finishers, and terrazzo workers'
 'Payroll and timekeeping clerks' 'Etchers and engravers'
 'Licensed practical and licensed vocational nurses' 'Teacher assistants'
 'Automotive service technicians and mechanics' 'Data entry keyers'
 'Miscellaneous healthcare support occupations, including medical equipment preparers'
 'File Clerks' 'Railroad conductors and yardmasters'
 'Power plant operators, distributors, and dispatchers'
 'Textile knitting and weaving machine setters, operators, and tenders'
 'Tailors, dressmakers, and sewers' 'Dental assistants'
 'Radio and telecommunications equipment installers and repairers'
 'Industrial engineers, including health and safety'
 'Medical, dental, and ophthalmic laboratory technicians'
 'Air traffic controllers and airfield operations specialists'
 'Forest and conservation workers' 'Roofers' 'Mechanical engineers'
 'Railroad brake, signal, and switch operators'
 'Interviewers, except eligibility and loan'
 'Sawing machine setters, operators, and tenders, wood'
 'Property, real estate, and community association managers'
 'Computer programmers'
 'Counter attendants, cafeteria, food concession, and coffee shop'
 'Veterinarians' 'Diagnostic related technologists and technicians'
 'Computer operators' 'Dispatchers'
 'Heating, air conditioning, and refrigeration mechanics and installers'
 'Parking lot attendants' 'Production, planning, and expediting clerks'
 'Recreation and fitness workers'
 'Door-to-door sales workers, news and street vendors, and related workers'
 'Engineering technicians, except drafters' 'Civil engineers'
 'Clinical laboratory technologists and technicians' 'Upholsterers'
 'Receptionists and information clerks' 'Social workers'
 'Library assistants, clerical'
 'Reservation and transportation ticket agents and travel clerks'
 'Counselors' 'Logging workers' 'Automotive body and related repairers'
 'Dishwashers' 'Librarians' 'Aircraft mechanics and service technicians'
 'Woodworking machine setters, operators, and tenders, except sawing'
 'Economists' 'Miscellaneous legal support workers'
 'Special education teachers'
 'Securities, commodities, and financial services sales agents' 'Lawyers'
 'Claims adjusters, appraisers, examiners, and investigators'
 'Administrative services managers' 'Designers'
 'Cabinetmakers and bench carpenters'
 'Shoe and leather workers and repairers' '.n:  No answer'
 'Textile, apparel, and furnishings workers, all other'
 'Other education, training, and library workers' 'Compliance officers'
 'Computer and information systems managers'
 'Brickmasons, blockmasons, and stonemasons'
 'Insurance claims and policy processing clerks'
 'Software developers, applications and systems software'
 'Animal trainers' 'Editors'
 'Combined food preparation and serving workers, including fast food'
 'Training and development specialists' 'Ship engineers' 'Travel agents'
 'Aerospace engineers' 'Telemarketers' 'Sailors and marine oilers'
 'Cleaning, washing, and metal pickling equipment operators and tenders'
 'Graders and sorters, agricultural products'
 'Mail clerks and mail machine operators, except postal service'
 'Metal furnace operators, tenders, pourers, and casters'
 'Tank car, truck, and ship loaders'
 'Dining room and cafeteria attendants and bartender helpers'
 'Heavy vehicle and mobile equipment service technicians and mechanics'
 'Meter readers, utilities'
 'Packaging and filling machine operators and tenders'
 'Architects, except naval' 'Sales representatives, services, all other'
 'General and operations managers'
 'First-line supervisors of non-retail sales workers'
 'Postmasters and mail superintendents'
 'Transportation, storage, and distribution managers'
 'Machine feeders and offbearers'
 'Miscellaneous life, physical, and social science technicians'
 'Bartenders' 'Medical and health services managers'
 'Television, video, and motion picture camera operators and'
 'Models, demonstrators, and product promoters'
 'Bus and truck mechanics and diesel engine specialists'
 'Billing and posting clerks'
 'Switchboard operators, including answering service'
 'Maintenance and repair workers, general' 'Flight attendants' 'Actors'
 'Bill and account collectors'
 'Miscellaneous community and social service specialists,'
 'Public relations and fundraising managers'
 'Eligibility interviewers, government programs' 'Physicians and surgeons'
 'Psychologists'
 'Postal service mail sorters, processors, and processing  machine operators'
 'Drilling and boring machine tool setters, operators, and tenders,  metal and plastic'
 '    operators, and tenders Furnace, kiln, oven, drier, and kettle operators and tenders'
 'Personal care aides' 'Personal financial advisors'
 'Taxi drivers and chauffeurs' 'Other extraction workers'
 'First-line supervisors of personal service workers'
 'Chefs and head cooks' 'Architectural and engineering managers'
 'Engine and other machine assemblers'
 'Sales and related workers, all other'
 'Agents and business managers of artists, performers,'
 'Layout workers, metal and plastic' 'Human resources managers'
 'First-line supervisors of protective service workers, all other'
 'Musicians, singers, and related workers'
 'Purchasing agents, except wholesale, retail, and farm products'
 'Dancers and choreographers' 'Credit authorizers, checkers, and clerks'
 'Statistical assistants'
 'Telecommunications  line installers and repairers'
 'Prepress technicians and workers'
 'Extruding, forming, pressing, and compacting machine setters,'
 'Mining machine operators' 'Nonfarm animal caretakers'
 'Marine engineers and naval architects'
 'Helpers, installation, maintenance, and repair workers'
 'Computer support specialists' 'Environmental engineers'
 'Transportation inspectors'
 'Rail-track laying and maintenance equipment operators'
 'Precision instrument and equipment repairers' 'Insurance underwriters'
 'Broadcast and sound engineering technicians and radio'
 'Proofreaders and copy markers'
 'Photographic process workers and processing machine operators'
 'Food servers, nonrestaurant' 'Home appliance repairers'
 'Advertising and promotions managers'
 'Material moving workers, all other' 'Pumping station operators'
 'Milling and planing machine setters, operators, and tenders, metal  and plastic'
 'Military officer special and tactical operations leaders'
 'Crossing guards' 'Environmental scientists and geoscientists'
 'Podiatrists' 'Library technicians'
 'Food and tobacco roasting, baking, and drying machine operators and tenders'
 'Food batchmakers' 'Engineers, all other'
 'First-line supervisors of farming, fishing, and forestry workers'
 'Bridge and lock tenders' 'Medical assistants'
 'Financial clerks, all other'
 'First-line supervisors of correctional officers'
 'Electronic equipment installers and repairers, motor vehicles'
 'Ship and boat captains and operators' 'Millwrights'
 'Counter and rental clerks' 'Miscellaneous plant and system operators'
 'Computer, automated teller, and office machine repairers'
 'Tool grinders, filers, and sharpeners'
 'Business operations specialists, all other'
 'Miscellaneous health technologists and technicians'
 'Religious workers, all other'
 'Supervisors of transportation and material moving workers'
 'Bailiffs, correctional officers, and jailers'
 'Water and wastewater treatment plant and system operators'
 'Physical therapist assistants and aides'
 'Miscellaneous entertainment attendants and related workers'
 'Molders and molding machine setters, operators, and tenders, metal and plastic'
 'Mining and geological engineers, including mining safety engineers'
 'Speech-language  pathologists'
 'Textile cutting machine setters, operators, and tenders'
 'Procurement clerks'
 'Derrick, rotary drill, and service unit operators, oil, gas, and mining'
 'Animal control workers'
 'Electronic home entertainment equipment installers and repairers'
 'Dredge, excavating, and loading machine operators'
 'Food processing workers, all other' 'Optometrists'
 'Chemists and materials scientists' 'Management analysts'
 'Credit counselors and loan officers' 'Dietitians and nutritionists'
 'First-line supervisors of housekeeping and janitorial workers'
 'Parking enforcement workers' 'First-line enlisted military supervisors'
 'Cost estimators' 'Surveyors, cartographers, and photogrammetrists'
 'Physical therapists' 'Gaming services workers'
 'Computer occupations, all other' 'Embalmers and funeral attendants'
 'Phlebotomists' 'Tax examiners and collectors, and revenue agents'
 'Helpers, production workers'
 'Electrical power-line installers and repairers' 'Petroleum engineers'
 'Photographers' 'Judges, magistrates, and other judicial workers'
 'Other installation, maintenance, and repair workers'
 'Septic tank servicers and sewer pipe cleaners'
 'Other transportation workers' 'Actuaries' 'Budget analysts'
 'Human resources assistants, except payroll and timekeeping'
 'Training and development managers'
 'First-line supervisors of landscaping, lawn service, and groundskeeping workers'
 'Adhesive bonding machine operators and tenders'
 'Helpers, extraction workers' 'Mine shuttle car operators'
 'News analysts, reporters and correspondents' 'Roof bolters, mining'
 'Molders, shapers, and casters, except metal and plastic'
 'Credit analysts' 'Mathematicians' 'Pharmacists' 'Nuclear engineers'
 'Computer hardware engineers'
 'Jewelers and precious stone and metal workers'
 'Agricultural and food science technicians' 'Furniture finishers'
 'Explosives workers, ordnance handling experts, and blasters'
 'Geological and petroleum technicians'
 'Ambulance drivers and attendants, except emergency medical  technicians'
 'Entertainers and performers, sports and related workers, all other'
 'Barbers' 'Boilermakers' 'Hoist and winch operators'
 'Reinforcing iron and rebar workers'
 'Health practitioner support technologists and technicians'
 'Insulation workers' 'Archivists, curators, and museum technicians'
 'Legislators' 'Athletes, coaches, umpires, and related workers'
 'Miscellaneous social scientists and related workers'
 'Pest control workers'
 'Model makers and patternmakers, metal and plastic'
 'Advertising sales agents' 'Respiratory therapists'
 'Coin, vending, and amusement machine servicers and repairers'
 'Parts salespersons'
 'Transportation attendants, except flight attendants'
 'Other healthcare practitioners and technical occupations'
 'Biological scientists' 'Information and record clerks, all other'
 'Pharmacy aides' 'Statisticians' 'Fishers and related fishing workers'
 'Fire inspectors' 'Roustabouts, oil and gas'
 'Miscellaneous vehicle and mobile equipment mechanics, installers, and repairers'
 'Subway, streetcar, and other rail transportation workers'
 'Food cooking machine operators and tenders'
 'Urban and regional planners' 'Private detectives and investigators'
 'Sociologists' 'Nurse anesthetists' 'Atmospheric and space scientists'
 'Announcers'
 'Rolling machine setters, operators, and tenders, metal and plastic'
 'Plasterers and stucco masons' 'Funeral service managers'
 'Aircraft pilots and flight engineers'
 'Ushers, lobby attendants, and ticket takers'
 'Construction and building inspectors'
 'Control and valve installers and repairers'
 'Medical records and health information technicians' 'Tax preparers'
 'Woodworkers, all other'
 'Market research analysts and marketing specialists'
 'Opticians, dispensing' 'Earth drillers, except oil and gas'
 'Agricultural and food scientists' 'Fish and game wardens'
 'Extruding and drawing machine setters, operators, and tenders,  metal and plastic'
 'Textile bleaching and dyeing machine operators and tenders'
 'Miscellaneous media and communication  workers'
 'Structural metal fabricators and fitters'
 'Morticians, undertakers, and funeral directors' 'Sales engineers'
 'First-line supervisors of fire fighting and prevention workers'
 'Detectives and criminal investigators' 'Loan interviewers and clerks'
 'Physician assistants' 'Nurse practitioners' 'Massage therapists'
 'Avionics technicians' 'Miscellaneous construction and related workers'
 'Biological technicians' 'Correspondence clerks'
 'Physical scientists, all other' 'Transportation security screeners'
 'Conveyor operators and tenders'
 'Veterinary assistants and laboratory animal caretakers'
 'Manufactured building and mobile home installers'
 'Computer and information research scientists'
 'Medical transcriptionists' 'Buyers and purchasing agents, farm products'
 'Lifeguards and other recreational, and all other protective service workers'
 'Residential advisors' 'Therapists, all other'
 'Conservation scientists and foresters' 'Occupational therapists'
 'Gaming managers' 'Computer control programmers and operators'
 'Small engine mechanics' 'Emergency management directors'
 'Locksmiths and safe repairers' 'Chiropractors' 'Paperhangers'
 'Operations research analysts' 'Financial specialists, all other'
 'Brokerage clerks' 'Database administrators'
 'Miscellaneous personal appearance workers'
 'Heat treating equipment setters, operators, and tenders, metal and plastic'
 'Paralegals and legal assistants' 'Technical writers'
 'Occupational therapy assistants and aides'
 'Probation officers and correctional treatment specialists'
 'Personal care and service workers, all other' 'Tour and travel guides'
 'Fundraisers' 'Network and computer systems administrators' 'Glaziers'
 'Cooling and freezing equipment operators and tenders'
 'Compensation and benefits managers'
 'First-line supervisors of police and detectives'
 'Model makers and patternmakers, wood'
 'Meeting, convention, and event planners' 'Materials engineers'
 'Forging machine setters, operators, and tenders, metal and plastic'
 'Automotive glass installers and repairers'
 'Emergency medical technicians and paramedics' 'Medical scientists'
 'Directors, religious activities and education'
 'First-line supervisors of gaming workers' 'Computer network architects'
 'Semiconductor processors'
 'Food preparation and serving related workers, all other'
 'Fabric and apparel patternmakers' 'Audiologists'
 'Information security analysts' 'Social science research assistants'
 'Astronomers and physicists' 'Financial examiners'
 'Signal and track switch repairers' 'Commercial divers'
 'New accounts clerks' 'Biomedical engineers'
 'Multiple machine tool setters, operators, and tenders, metal and plastic'
 'Agricultural inspectors'
 'Extruding and forming machine setters, operators, and tenders,  synthetic and glass fibers'
 'Transit and railroad police' 'Hazardous materials removal workers'
 'Health diagnosing and treating practitioners, all other'
 'Judicial law clerks' 'Recreational therapists'
 'Cargo and freight agents' 'Natural sciences managers'
 'Exercise physiologists' 'Logisticians' 'Desktop publishers'
 'Motor vehicle operators, all other'
 'Security and fire alarm systems installers' 'Web developers'
 'Nuclear technicians' 'Riggers' 'Animal breeders' 'Fence erectors'
 'Paving, surfacing, and tamping equipment operators' 'Survey researchers'
 'Radiation therapists' 'Miscellaneous mathematical science occupations'
 '.s:  Skipped on Web' 'Nurse midwives' 'Agricultural engineers'
 'Hunters and trappers']
sphrs1: ['.i:  Inapplicable' '40' '37' '51' '60' '45' '44' '55' '24' '80' '65'
 '50' '75' '16' '36' '72' '15' '62' '20' '70' '63' '48' '19' '30' '56'
 '22' '38' '84' '18' '35' '52' '57' '32' '58' '43' '25' '74'
 '.n:  No answer' '12' '3' '42' '4' '8' '5'
 '.d:  Do not Know/Cannot Choose' '10' '66' '28' '89+ hrs' '39' '33' '59'
 '11' '54' '21' '79' '9' '46' '69' '6' '29' '47' '23' '27' '68' '13' '73'
 '34' '76' '49' '14' '78' '41' '64' '53' '77' '82' '26' '85' '2' '61' '88'
 '31' '7' '17' '71' '0' '67' '81' '1' '86' '.s:  Skipped on Web']
sphrs2: ['.i:  Inapplicable' '73' '20' '.d:  Do not Know/Cannot Choose' '40' '35'
 '48' '14' '50' '.n:  No answer' '16' '84' '44' '56' '13' '32' '8' '37'
 '72' '24' '12' '60' '54' '22' '43' '25' '38' '49' '30' '45' '36' '65'
 '57' '89+ hrs' '55' '52' '70' '51' '21' '26' '10' '5' '46' '80' '47' '1'
 '42' '39' '33' '27' '18' '15' '34' '.s:  Skipped on Web' '9']
happy: ['Not too happy' 'Pretty happy' 'Very happy' '.n:  No answer'
 '.d:  Do not Know/Cannot Choose' '.i:  Inapplicable'
 '.s:  Skipped on Web']
hapmar: ['.i:  Inapplicable' 'VERY HAPPY' 'PRETTY HAPPY' 'NOT TOO HAPPY'
 '.n:  No answer' '.d:  Do not Know/Cannot Choose' '.s:  Skipped on Web']
joblose: ['.i:  Inapplicable' 'Not likely' 'Not too likely' 'Very likely'
 'Fairly likely' '.d:  Do not Know/Cannot Choose' 'Leaving labor force'
 '.n:  No answer' '.s:  Skipped on Web']
satjob: ['A little dissatisfied' '.i:  Inapplicable' 'Moderately satisfied'
 'Very satisfied' 'Very dissatisfied' '.n:  No answer'
 '.d:  Do not Know/Cannot Choose' '.s:  Skipped on Web']
class_: ['Middle class' 'Working class' 'Upper class' 'Lower class'
 '.n:  No answer' '.i:  Inapplicable' '.d:  Do not Know/Cannot Choose'
 'No class' '.s:  Skipped on Web']
satfin: ['Not satisfied at all' 'More or less satisfied' 'Pretty well satisfied'
 '.n:  No answer' '.d:  Do not Know/Cannot Choose' '.i:  Inapplicable'
 '.s:  Skipped on Web']
finalter: ['Better' 'Stayed same' 'Worse' '.d:  Do not Know/Cannot Choose'
 '.n:  No answer' '.i:  Inapplicable' '.s:  Skipped on Web']
tvhours: ['.i:  Inapplicable' '1' '2' '3' '4' '5' '6' '0 hours' '8' '10' '12' '14'
 '7' '.n:  No answer' '15' '9' '17' '13' '16' '11' '20' '24 hours'
 '.d:  Do not Know/Cannot Choose' '18' '21' '22' '23'
 '.s:  Skipped on Web' '19']
wrktype: ['.i:  Inapplicable' 'Regular, permanent employee'
 'Independent contractor/consultant/freelance worker'
 'Work for contractor who provides workers/services' '.n:  No answer'
 'On-call, work only when called to work' '.d:  Do not Know/Cannot Choose'
 'Paid by a temporary agency' '.s:  Skipped on Web']
yearsjob: [-100.0 '-100.0' '1.0' '2.0' '3.0' '5.0' '6-11.9 months'
 'Less than 6 months' '15.0' '12.0' '6.0' '16.0' '17.0' '4.0' '32.0' '7.0'
 '30.0' '-99.0' '22.0' '9.0' '29.0' '20.0' '8.0' '23.0' '11.0' '-98.0'
 '10.0' '14.0' '21.0' '13.0' '27.0' '28.0' '18.0' '26.0' '24.0' '25.0'
 '40.0' '36.0' '31.0' '42.0' '19.0' '47.0' '34.0' '33.0' '43.0' '48.0'
 '37.0' '50.0' '45.0' '35.0' '38.0' '39.0' '60.0' '46.0' '57.0' '44.0'
 '41.0' '55.0' '67.0' '51.0' '49.0' '-97.0' '58.0' '56.0']
waypaid: ['.i:  Inapplicable' 'Paid by the hour' 'Salaried' 'Other'
 '.n:  No answer' '.d:  Do not Know/Cannot Choose' 'Self-employed'
 'Commission' 'Paid by the job' 'Salary and bonus/commission' 'Tips'
 'Paid by percent' 'Paid by contract' 'Paid by day' 'Piece work'
 '.s:  Skipped on Web']
wrksched: ['.i:  Inapplicable' 'Day shift' 'Rotating shifts' 'Afternoon shift'
 'Irregular shift/on-call' 'Split shift' '.n:  No answer' 'Night shift'
 '.d:  Do not Know/Cannot Choose' '.s:  Skipped on Web']
moredays: ['.i:  Inapplicable' '5' '8' '2' '10' '6' '0' '4' '3' '20' '24'
 '.d:  Do not Know/Cannot Choose' '1' '30' '.n:  No answer' '21' '25' '7'
 '18' '15' '22' '12' '16' '19' '11' '17' '9' '28' '23' '27' '14' '26' '13'
 '29' '-97' '31']
mustwork: ['.i:  Inapplicable' 'NO' 'YES' '.n:  No answer'
 '.d:  Do not Know/Cannot Choose' '.s:  Skipped on Web']
wrkhome: ['.i:  Inapplicable' 'Never' 'About once a month' 'A few times a year'
 'More than once a week' 'Worker works mainly at home' 'About once a week'
 '.n:  No answer' '.d:  Do not Know/Cannot Choose' '.s:  Skipped on Web']
whywkhme: ['.i:  Inapplicable' 'Worker wants to work at home'
 'Worker has to work at home to keep up with job'
 'Worker is operating home-based business' '.n:  No answer'
 '.d:  Do not Know/Cannot Choose' 'Other combinations and other reasons'
 '.s:  Skipped on Web']
famwkoff: ['.i:  Inapplicable' 'Not at all hard' 'Not too hard' 'Very hard'
 'Somewhat hard' '.n:  No answer' '.d:  Do not Know/Cannot Choose'
 '.s:  Skipped on Web']
wkvsfam: ['.i:  Inapplicable' 'Rarely' 'Never' 'Often' 'Sometimes' '.n:  No answer'
 '.d:  Do not Know/Cannot Choose' '.s:  Skipped on Web']
famvswk: ['.i:  Inapplicable' 'Rarely' 'Sometimes' 'Never' 'Often' '.n:  No answer'
 '.d:  Do not Know/Cannot Choose' '.s:  Skipped on Web']
hrsrelax: ['.i:  Inapplicable' '3' '2' '1' '0' '6' '4' '5' '.n:  No answer' '12' '7'
 '10' '.d:  Do not Know/Cannot Choose' '8' '15' '11' '24' '20' '9' '14'
 '16' '19' '21' '18' '22' '13' '17' '-97']
secondwk: ['.i:  Inapplicable' 'YES' 'NO' '.n:  No answer'
 '.d:  Do not Know/Cannot Choose' '.s:  Skipped on Web']
learnnew: ['.i:  Inapplicable' 'Agree' 'Strongly Agree' 'Disagree'
 'Strongly Disagree' '.n:  No answer' '.d:  Do not Know/Cannot Choose'
 '.s:  Skipped on Web']
workfast: ['.i:  Inapplicable' 'Agree' 'Disagree' 'Strongly Agree' '.n:  No answer'
 '.d:  Do not Know/Cannot Choose' 'Strongly Disagree'
 '.s:  Skipped on Web']
overwork: ['.i:  Inapplicable' 'Disagree' 'Agree' 'Strongly Agree'
 'Strongly Disagree' '.n:  No answer' '.d:  Do not Know/Cannot Choose'
 '.s:  Skipped on Web']
respect: ['.i:  Inapplicable' 'Strongly Agree' 'Agree' 'Disagree' '.n:  No answer'
 '.d:  Do not Know/Cannot Choose' 'Strongly Disagree'
 '.s:  Skipped on Web']
trustman: ['.i:  Inapplicable' 'Strongly Agree' 'Agree' 'Strongly Disagree'
 'Disagree' '.d:  Do not Know/Cannot Choose' '.n:  No answer'
 '.s:  Skipped on Web']
proudemp: ['.i:  Inapplicable' 'Strongly Agree' 'Agree' 'Disagree'
 '.d:  Do not Know/Cannot Choose' '.n:  No answer' 'Strongly Disagree'
 '.s:  Skipped on Web']
supcares: ['.i:  Inapplicable' 'Very true' 'Somewhat true' 'Not at all true'
 '.n:  No answer' '.d:  Do not Know/Cannot Choose' 'Not too true'
 '.s:  Skipped on Web']
condemnd: ['.i:  Inapplicable' 'Very true' 'Somewhat true' 'Not at all true'
 'Not too true' '.d:  Do not Know/Cannot Choose' '.n:  No answer'
 '.s:  Skipped on Web']
promtefr: ['.i:  Inapplicable' 'Very true' 'Somewhat true' 'Not at all true'
 'Not too true' '.d:  Do not Know/Cannot Choose' '.n:  No answer'
 '.s:  Skipped on Web']
cowrkint: ['.i:  Inapplicable' 'Very true' 'Somewhat true' 'Not too true'
 'Not at all true' '.d:  Do not Know/Cannot Choose' '.n:  No answer'
 '.s:  Skipped on Web']
jobsecok: ['.i:  Inapplicable' 'Very true' 'Somewhat true' 'Not at all true'
 'Not too true' '.d:  Do not Know/Cannot Choose' '.n:  No answer'
 '.s:  Skipped on Web']
manvsemp: ['.i:  Inapplicable' 'Very good' 'Quite good' 'Neither good nor bad'
 'Quite bad' '.d:  Do not Know/Cannot Choose' '.n:  No answer' 'Very bad'
 '.s:  Skipped on Web']
trynewjb: ['.i:  Inapplicable' 'Not at all likely' 'Very likely' 'Somewhat likely'
 '.n:  No answer' '.d:  Do not Know/Cannot Choose' '.s:  Skipped on Web']
health1: ['.i:  Inapplicable' 'Excellent' 'Good' 'Very good' 'Poor' 'Fair'
 '.d:  Do not Know/Cannot Choose' '.n:  No answer' '.s:  Skipped on Web']
mntlhlth: ['.i:  Inapplicable' '0' '2' '30' '5' '19' '15' '10' '3' '7' '1'
 '.n:  No answer' '4' '25' '8' '.d:  Do not Know/Cannot Choose' '20' '22'
 '6' '14' '9' '12' '18' '28' '21' '26' '23' '11' '17' '29' '16' '24' '27'
 '-97' '13']
spvtrfair: ['.i:  Inapplicable' 'Very true' 'Not too true' 'Somewhat true'
 '.d:  Do not Know/Cannot Choose' 'Not at all true' '.n:  No answer'
 '.s:  Skipped on Web']
slpprblm: ['.i:  Inapplicable' 'Rarely' 'Often' 'Never' 'Sometimes' '.n:  No answer'
 '.d:  Do not Know/Cannot Choose' '.s:  Skipped on Web']
satjob1: ['.i:  Inapplicable' 'Very satisfied' 'Somewhat satisfied'
 'Not too satisfied' 'Not at all satisfied' '.n:  No answer'
 '.d:  Do not Know/Cannot Choose' '.s:  Skipped on Web']
hyperten: ['.i:  Inapplicable' 'No' 'Yes' '.n:  No answer'
 '.d:  Do not Know/Cannot Choose' '.s:  Skipped on Web']
stress: ['.y:  Not available in this year' '.i:  Inapplicable' '.n:  No answer'
 'Sometimes' 'Often' 'Hardly ever' '.d:  Do not Know/Cannot Choose'
 'Never' 'Always' '.s:  Skipped on Web']
realinc: [ 1.89510000e+04  2.43660000e+04  3.04580000e+04  5.07630000e+04
  4.39940000e+04  3.72260000e+04  1.35370000e+04  2.70700000e+03
  6.09150000e+04 -1.00000000e+02  7.44520000e+04  8.12200000e+03
  1.09355000e+05  3.27610000e+04  1.96570000e+04  1.17940000e+04
  5.24200000e+03  4.58660000e+04  1.44150000e+04  5.89700000e+04
  9.17300000e+03  1.31000000e+03  2.35880000e+04  9.32100000e+04
  1.70360000e+04  8.63600000e+03  4.31780000e+04  4.93500000e+03
  2.22060000e+04  3.08410000e+04  5.55150000e+04  1.11030000e+04
  1.35700000e+04  9.47380000e+04  1.23400000e+03  1.85050000e+04
  1.60380000e+04  1.44520000e+04  2.77930000e+04  1.22290000e+04
  5.00270000e+04  3.89100000e+04  8.64960000e+04  2.00110000e+04
  1.00050000e+04  1.66760000e+04  7.78200000e+03  4.44700000e+03
  1.11200000e+03  7.93400000e+04  3.56510000e+04  4.58370000e+04
  1.01900000e+03  1.52790000e+04  2.54650000e+04  7.13000000e+03
  4.07400000e+03  9.16700000e+03  1.12050000e+04  1.32420000e+04
  1.83350000e+04  1.44460000e+04  2.64840000e+04  4.09300000e+04
  7.22290000e+04  3.12990000e+04  6.53586240e+04  8.66700000e+03
  2.16690000e+04  1.25200000e+04  3.85200000e+03  3.61140000e+04
  1.05940000e+04  1.62607000e+05  1.73350000e+04  6.74100000e+03
  9.63000000e+02  2.48790000e+04  6.33300000e+03  2.94020000e+04
  5.75212890e+04  6.78510000e+04  1.43778000e+05  3.39260000e+04
  1.62840000e+04  2.03550000e+04  3.84490000e+04  3.61900000e+03
  1.17610000e+04  9.05000000e+02  9.95200000e+03  8.14200000e+03
  8.30800000e+03  1.69940000e+04  3.02100000e+03  2.07700000e+04
  1.28141000e+05  2.83230000e+04  3.21000000e+04  2.45470000e+04
  5.66470000e+04  5.25101445e+04  9.81900000e+03  1.35950000e+04
  6.79800000e+03  5.28700000e+03  1.13290000e+04  7.55000000e+02
  2.26050000e+04  4.31528415e+04  7.83600000e+03  4.22000000e+03
  1.95910000e+04  5.12370000e+04  1.35630000e+04  2.41100000e+03
  3.61670000e+04  9.07220000e+04  2.56190000e+04  1.08500000e+04
  6.03000000e+02  5.42500000e+03  1.65770000e+04  9.04200000e+03
  6.63100000e+03  6.24800000e+03  4.06828935e+04  1.10895000e+05
  3.40780000e+04  1.02230000e+04  2.41390000e+04  4.82770000e+04
  1.84590000e+04  2.27200000e+03  2.12990000e+04  1.27790000e+04
  1.56190000e+04  5.11200000e+03  3.97600000e+03  8.52000000e+03
  7.38400000e+03  5.68000000e+02  4.67730000e+04  1.51320000e+04
  9.28580000e+04  3.30160000e+04  1.23810000e+04  2.33860000e+04
  1.78840000e+04  3.90204285e+04  6.05300000e+03  9.90500000e+03
  2.06350000e+04  7.15300000e+03  8.25400000e+03  2.20100000e+03
  4.95200000e+03  5.50000000e+02  3.16680000e+04  9.92570000e+04
  3.69500000e+03  1.18760000e+04  1.45150000e+04  2.24320000e+04
  6.86100000e+03  5.80600000e+03  3.75004605e+04  1.71540000e+04
  9.50000000e+03  4.48630000e+04  4.75000000e+03  1.97930000e+04
  5.28000000e+02  7.91700000e+03  2.11100000e+03  2.16590000e+04
  5.60580000e+04  2.80290000e+04  5.60600000e+03  1.14660000e+04
  1.40150000e+04  7.64400000e+03  4.58700000e+03  6.62500000e+03
  3.31250000e+04  3.82220000e+04  2.42070000e+04  1.02084000e+05
  1.91110000e+04  1.65630000e+04  2.03800000e+03  3.56700000e+03
  5.10000000e+02  9.02780000e+04  3.50000000e+03  2.75000000e+04
  4.50000000e+04  5.50000000e+04  3.25000000e+04  4.50000000e+03
  1.12500000e+04  3.75000000e+04  2.12500000e+04  7.50000000e+03
  1.37500000e+04  1.87500000e+04  2.37500000e+04  1.62500000e+04
  9.00000000e+03  2.00000000e+03  5.00000000e+02  6.50000000e+03
  5.50000000e+03  9.15870000e+04  3.13640000e+04  1.32690000e+04
  4.34260000e+04  3.61890000e+04  2.29190000e+04  3.37800000e+03
  2.65380000e+04  6.27300000e+03  1.08570000e+04  8.68500000e+03
  1.80940000e+04  4.83000000e+02  7.23800000e+03  2.05070000e+04
  5.30770000e+04  4.34300000e+03  1.56820000e+04  5.30800000e+03
  1.93000000e+03  8.81330000e+04  8.34900000e+03  1.04360000e+04
  1.97130000e+04  6.03000000e+03  1.50750000e+04  2.20320000e+04
  1.73940000e+04  2.55110000e+04  4.17460000e+04  5.10230000e+04
  1.85500000e+03  1.27560000e+04  3.01500000e+04  3.24700000e+03
  4.64000000e+02  3.47880000e+04  6.95800000e+03  5.10200000e+03
  4.17500000e+03  2.88300000e+04  7.98400000e+03  9.99880000e+04
  3.99200000e+03  2.43950000e+04  9.98000000e+03  3.99190000e+04
  6.65300000e+03  3.10500000e+03  4.87900000e+04  1.88510000e+04
  3.32660000e+04  1.66330000e+04  1.77400000e+03  2.10680000e+04
  5.76600000e+03  4.87900000e+03  1.21970000e+04  4.44000000e+02
  6.26700000e+03  9.02650000e+04  9.40100000e+03  4.59600000e+03
  5.43200000e+03  7.52100000e+03  1.98460000e+04  1.77570000e+04
  2.29800000e+04  3.13360000e+04  3.76030000e+04  1.35790000e+04
  3.76000000e+03  2.71580000e+04  1.14900000e+04  4.59590000e+04
  5.64050000e+04  1.56680000e+04  1.67100000e+03  2.92500000e+03
  4.18000000e+02  3.53700000e+04  1.03880000e+05  1.67030000e+04
  4.32300000e+04  1.08080000e+04  8.84300000e+03  2.55450000e+04
  2.85706485e+04  4.32300000e+03  5.30550000e+04  2.16150000e+04
  7.07400000e+03  2.94750000e+04  1.27730000e+04  3.93000000e+02
  1.57200000e+03  1.47380000e+04  3.53700000e+03  5.10900000e+03
  5.89500000e+03  2.75100000e+03  8.59500000e+03  1.24150000e+04
  1.43250000e+04  1.05050000e+04  3.43800000e+04  2.78819130e+04
  2.10100000e+04  2.48300000e+04  1.00963265e+05  5.15700000e+04
  4.20200000e+04  4.20200000e+03  2.67400000e+03  6.87600000e+03
  4.96600000e+03  1.52800000e+03  3.43800000e+03  1.62350000e+04
  2.86500000e+04  5.73000000e+03  3.82000000e+02  3.26250000e+04
  6.52500000e+03  1.35940000e+04  9.96900000e+03  1.54060000e+04
  3.98750000e+04  4.89380000e+04  3.63000000e+02  2.35630000e+04
  3.98800000e+03  2.71880000e+04  1.17810000e+04  1.99380000e+04
  1.02565000e+05  5.43800000e+03  2.66469390e+04  8.15600000e+03
  3.26300000e+03  1.45000000e+03  4.71300000e+03  2.53800000e+03
  3.10050000e+04  1.11960000e+04  5.68420000e+04  1.63630000e+04
  6.89000000e+04  4.65070000e+04  3.78950000e+04  1.89470000e+04
  1.46410000e+04  7.75100000e+03  3.45000000e+02  1.15841000e+05
  2.58370000e+04  2.23920000e+04  6.20100000e+03  5.16700000e+03
  1.29180000e+04  4.47800000e+03  9.47300000e+03  3.78900000e+03
  1.37800000e+03  3.10000000e+03  5.98500000e+03  1.24680000e+04
  2.99250000e+04  2.49370000e+04  1.82870000e+04  1.41310000e+04
  1.41038000e+05  6.65000000e+04  3.65750000e+04  4.48870000e+04
  9.14300000e+03  1.57930000e+04  1.33000000e+03  5.48620000e+04
  2.16120000e+04  7.48100000e+03  1.08060000e+04  3.33000000e+02
  3.65700000e+03  4.98700000e+03  2.32700000e+03  2.99200000e+03
  4.32200000e+03  2.03352500e+04  1.37237766e+05  2.34637500e+04
  2.81565000e+04  6.25700000e+04  4.22347500e+04  3.44135000e+04
  5.16202500e+04  1.32961250e+04  2.18995000e+03  1.01676250e+04
  3.12850000e+02  1.72067500e+04  3.44135000e+03  8.60337500e+03
  1.48603750e+04  7.03912500e+03  1.17318750e+04  4.06705000e+03
  5.63130000e+03  2.81565000e+03  1.25140000e+03  4.69275000e+03
  3.32090000e+04  3.32090000e+03  1.28434606e+05  2.26425000e+04
  1.13212500e+04  1.96235000e+04  1.20760000e+03  4.07565000e+04
  3.01900000e+02  1.28307500e+04  4.98135000e+04  1.66045000e+04
  2.11330000e+03  2.71710000e+04  8.30225000e+03  6.03800000e+04
  5.43420000e+03  9.81175000e+03  1.43402500e+04  6.79275000e+03
  2.71710000e+03  4.52850000e+03  3.92470000e+03  1.84762500e+04
  4.69012500e+04  2.55825000e+04  2.13187500e+04  3.12675000e+04
  5.11650000e+03  6.39562500e+03  2.55825000e+03  1.56337500e+04
  1.20806250e+04  3.83737500e+04  3.69525000e+03  6.82200000e+04
  1.44502717e+05  1.06593750e+04  7.95900000e+04  5.68500000e+04
  4.26375000e+03  9.23812500e+03  3.12675000e+03  1.35018750e+04
  2.84250000e+02  7.81687500e+03  1.98975000e+03  1.13700000e+03
  4.01625000e+03  6.42600000e+04  2.40975000e+04  2.00812500e+04
  1.74037500e+04  1.46153669e+05  1.13793750e+04  4.41787500e+04
  3.61462500e+04  2.94525000e+04  5.35500000e+04  1.47262500e+04
  6.02437500e+03  7.49700000e+04  1.27181250e+04  4.81950000e+03
  8.70187500e+03  1.00406250e+04  2.67750000e+02  7.36312500e+03
  1.07100000e+03  1.87425000e+03  2.94525000e+03  3.48075000e+03
  2.40975000e+03  4.27350000e+04  8.41750000e+03  1.10075000e+04
  4.66200000e+03  5.82750000e+03  2.84900000e+04  2.33100000e+04
  5.18000000e+04  1.23025000e+04  2.33100000e+03  7.12250000e+03
  3.49650000e+04  1.81300000e+03  1.19606065e+05  2.84900000e+03
  7.25200000e+04  9.71250000e+03  1.42450000e+04  6.21600000e+04
  1.68350000e+04  2.59000000e+02  1.94250000e+04  1.03600000e+03
  3.88500000e+03  3.36700000e+03  1.55139973e+05  5.88000000e+04
  6.86000000e+04  2.69500000e+04  1.34750000e+04  1.59250000e+04
  2.69500000e+03  2.45000000e+02  4.41000000e+03  1.16375000e+04
  2.20500000e+04  3.30750000e+04  4.90000000e+04  4.04250000e+04
  1.83750000e+04  7.96250000e+03  1.04125000e+04  3.18500000e+03
  9.18750000e+03  6.73750000e+03  5.51250000e+03  9.80000000e+02
  3.67500000e+03  2.20500000e+03  1.71500000e+03  3.90225000e+04
  1.34817440e+05  2.12850000e+04  7.68625000e+03  4.73000000e+04
  5.32125000e+03  8.86875000e+03  1.12337500e+04  1.30075000e+04
  3.19275000e+04  3.54750000e+03  1.53725000e+04  2.36500000e+02
  5.67600000e+04  2.60150000e+04  1.77375000e+04  3.07450000e+03
  1.00512500e+04  1.65550000e+03  6.62200000e+04  6.50375000e+03
  4.25700000e+03  2.60150000e+03  9.46000000e+02  2.12850000e+03
  1.31676691e+05  2.57400000e+04  3.86100000e+04  3.15900000e+04
  1.52100000e+04  2.34000000e+02  4.21200000e+03  6.43500000e+03
  2.10600000e+04  2.57400000e+03  1.75500000e+04  1.28700000e+04
  4.68000000e+04  1.11150000e+04  9.94500000e+03  5.61600000e+04
  7.48800000e+04  6.55200000e+04  9.36000000e+02  8.77500000e+03
  3.04200000e+03  5.26500000e+03  7.60500000e+03  3.51000000e+03
  1.63800000e+03  2.10600000e+03  1.47550000e+04  7.26400000e+04
  1.19879417e+05  8.51250000e+03  2.49700000e+04  2.49700000e+03
  4.54000000e+04  9.64750000e+03  1.07825000e+04  5.44800000e+04
  9.08000000e+02  3.06450000e+04  3.74550000e+04  1.70250000e+04
  5.10750000e+03  6.35600000e+04  2.04300000e+04  6.24250000e+03
  1.24850000e+04  4.08600000e+03  3.40500000e+03  7.37750000e+03
  2.27000000e+02  2.04300000e+03  1.58900000e+03  2.95100000e+03
  1.20197019e+05  4.36000000e+04  2.94300000e+04  3.92400000e+03
  3.59700000e+04  1.96200000e+04  5.23200000e+04  9.26500000e+03
  1.19900000e+04  6.97600000e+04  2.39800000e+04  1.03550000e+04
  6.10400000e+04  4.90500000e+03  7.08500000e+03  1.41700000e+04
  5.99500000e+03  1.63500000e+04  2.18000000e+02  8.72000000e+02
  8.17500000e+03  2.83400000e+03  1.96200000e+03  1.52600000e+03
  3.27000000e+03  2.39800000e+03  4.09000000e+04  1.84050000e+04
  1.12475000e+04  2.24950000e+04  1.53375000e+04  1.35195964e+05
  9.71375000e+03  4.60125000e+03  1.32925000e+04  5.72600000e+04
  8.18000000e+02  6.64625000e+03  5.62375000e+03  3.37425000e+04
  2.04500000e+02  6.54400000e+04  4.90800000e+04  2.24950000e+03
  2.65850000e+03  7.66875000e+03  3.68100000e+03  2.76075000e+04
  8.69125000e+03  3.06750000e+03  1.43150000e+03  1.84050000e+03]
ballot: ['.i:  Inapplicable' 'Ballot b' 'Ballot c' 'Ballot a' 'Ballot d']
sei10: [  50.    46.5   56.9   76.3   31.9   62.    36.2   32.    13.6   12.6
   12.4   29.5   39.9   36.4   41.1   82.5 -100.    59.1   60.4   38.8
   80.9   14.8   39.7   38.1   21.    28.4   27.7   84.2   50.4   32.7
   84.5   24.6   74.6   28.7   19.7   23.    31.1   42.8   20.8   25.
   35.8   18.6   14.    26.8   13.2   52.4   38.3   35.1   29.2   19.6
   37.3   19.2   81.9   87.6   65.3   24.2   39.1   69.3   43.    26.2
   23.2   25.2   60.1   24.1   62.6   20.7   22.9   62.7   53.5   45.4
   48.8   48.1   31.    45.3   39.6   80.7   77.4   21.6   23.3   27.
   71.6   72.7   76.7   75.5   52.    43.8   37.6   58.4   15.8   27.2
   21.8   73.9   68.    29.7    9.    92.1   50.3   34.1   59.8   67.7
   65.1   18.8   31.6   25.1   23.7   48.3   73.6   49.6   62.4   35.2
   22.4   86.5   57.1   17.5   37.7   24.    57.8   67.8   20.9   11.8
   46.    34.7   27.5   53.8   23.8   41.    21.4   41.8   46.6   26.1
   31.2   32.6   28.6   35.3   58.1   65.2   12.7   26.5   52.1   81.6
   37.2   66.4   31.5   20.5   83.8   14.4   54.6   81.    14.6   89.9
   47.9   38.2   22.1   49.4   42.9   57.4   85.2   60.5   20.3   25.7
   62.9   36.5   44.9   63.7   20.    28.3   10.6   77.2   53.6   17.1
   91.9   56.8   71.7   71.    91.1   64.9   61.4   27.1   26.6   29.9
   70.3   69.4   39.2   87.9   37.4   73.4   64.2   40.1   45.    89.3
   30.3   19.    33.7   31.3   13.3   37.    84.    61.2   68.6   57.5
   52.5   21.1   55.1   26.9   71.8   38.9   33.4   52.6   55.    34.8
   82.4   52.9   92.8   88.7   58.2   39.8   20.1   79.3   28.5   36.7
   89.1   34.4   55.3   60.2   52.3   59.9   39.5   46.2   34.6   24.8
   42.2   82.9   64.5   85.5   50.1   40.    53.9   66.2   57.3   34.5
   17.    33.9   73.1   43.4   70.1   86.7   88.3   27.4   86.1   41.6
   50.7   54.3   39.    30.9   53.2   51.8   25.5   49.2   51.7   45.1
   43.5   58.6   29.4   84.3   53.    30.1   91.6   86.6   78.3   62.1
   33.2   63.1   77.9   87.3   37.1   70.5   41.2   30.    60.3   51.9
   51.5   35.4   31.4   93.7   77.3   44.6   22.5   72.5   65.6   85.
   92.    81.5   44.5   63.2   26.7   49.3   21.5   33.1   25.6   27.6
   70.4   59.4   82.7   32.8   61.5   79.9   29.6   33.3   85.8   43.1
   35.5   30.8   59.    21.3   48.4   18.9   63.3   88.6   85.9   53.7
   25.4   79.4   36.9   57.    28.8   75.6   41.4   77.7   28.1   66.1
   75.2   73.5   42.7   75.7   92.2   37.8   53.1   54.1   86.8   21.2
   37.9   36.8   29.3   71.4   80.1   89.4   38.4   83.4   80.2   64.4
   77.    32.1   71.1   40.4   73.    74.8   73.2   72.6   83.1   88.
   46.7   88.4   87.8   76.5   51.6   69.5   79.6   56.3   86.9   61.9
   42.3   86. ]

    """