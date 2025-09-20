import streamlit as st
import pandas as pd
from collections import Counter
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import PyPDF2
import re
import random
from collections import Counter

# -------------------------------
# Define categories and phrases
# -------------------------------
CLASSES = [
    'Promotional', 'Social', 'Financial', 'Educational',
    'Healthcare', 'Shopping', 'Travel', 'Spam','Job Alerts','Technology','Government',
    'News & Updates',
    'Events',
    'Legal',
    'Support/Service'
]

ALL_PHRASES = {
    'Promotional': [
        'Limited-time offer: Get 40% off today.',
        'Exclusive coupon available in your account.',
        'Buy one, get one free this weekend.',
        'Hurry, mega clearance sale ends soon.',
        'Your reward points are about to expire.',
        'Sign up today and receive a 30% discount.',
        'Special holiday deals are available now.',
        'Unlock exclusive discounts with our app.',
        'Festival offer: Save up to 70% on electronics.',
        'Flash sale: Flat 50% off on fashion items.',
        'Diwali special: Grab your discount now.',
        'Free gift with your annual subscription.',
        'New season sale: Fresh arrivals at low prices.',
        'Save more with our loyalty program.',
        'Early bird discount ends tomorrow.',
        'Extra 10% off with coupon code inside.',
        'Special bundle price for a limited time.',
        'Free shipping on orders above Rs.500.',
        'Exclusive app-only offer: Download to claim.',
        'Clearance: Last chance to shop at lowest prices.'
    ],
    'Social': [
        'New friend request received.',
        'You have a new follower on Instagram.',
        'A friend tagged you in a photo.',
        'Someone commented on your post.',
        'You have received a new private message.',
        'See what your network is sharing today.',
        'Rahul mentioned you in a story.',
        'Your LinkedIn connection request was accepted.',
        'Reminder: Zoom meeting starts at 5 PM.',
        'An invitation to join a group chat.',
        'Your WhatsApp verification code is 123456.',
        'Sneha shared a video with you.',
        'Your post received 100 new likes.',
        'You have a new invitation to connect.',
        'You have been added to a new channel.',
        'Google Meet session reminder for tomorrow.',
        'Invitation to join a Discord community call.',
        'Your friend shared a new photo album.',
        'You were tagged in a group message.',
        'Congratulations, you earned a new badge.'
    ],
    'Financial': [
        'Your bank statement for this month is ready.',
        'Loan EMI of Rs.5000 is due tomorrow.',
        'Salary of Rs.40,000 has been credited.',
        'Your mutual fund SIP was processed successfully.',
        'Credit card bill payment reminder.',
        'Transaction alert: Rs.2500 debited from your account.',
        'Insurance premium payment is due this week.',
        'Fixed deposit maturity date is approaching.',
        'New credit card offer: 5% cashback on dining.',
        'Tax payment receipt is now available.',
        'UPI transaction of Rs.750 was successful.',
        'Investment summary for Q2 is ready.',
        'Your pension statement is available for download.',
        'Debit card transaction alert for Rs.3000.',
        'Mobile recharge of Rs.399 completed successfully.',
        'Forex purchase: USD 500 completed.',
        'Bank loan approved; disbursal in progress.',
        'Electricity bill of Rs.1500 is pending.',
        'PayPal payment of $100 received.',
        'Credit card reward points have been updated.'
    ],
    'Educational': [
        'Semester results have been published.',
        'Reminder: Submit your assignment by Friday.',
        'New course material uploaded to the portal.',
        'Data Science class starts tomorrow.',
        'Certificate of completion is ready for download.',
        'Exam schedule for December is available.',
        'Invitation to participate in a coding competition.',
        'Your internship application has been approved.',
        'Online quiz on Machine Learning is live.',
        'Library book return due in three days.',
        'Workshop on Cloud Computing is scheduled this week.',
        'Attendance report for semester five is ready.',
        'Congratulations, you have been awarded a scholarship.',
        'New study material for Physics is available.',
        'Feedback form submission is still pending.',
        'Join the webinar on Artificial Intelligence.',
        'Course subscription renewed successfully.',
        'Practical exam starts next Monday.',
        'Midterm results are now available.',
        'Project submission deadline has been extended.'
    ],
    'Healthcare': [
        'Your appointment with Dr. Mehta is confirmed.',
        'Annual health checkup reminder.',
        'Lab test results are available online.',
        'Prescription ready for pickup at your pharmacy.',
        'Reminder: Doctor appointment scheduled for tomorrow.',
        'Health insurance renewal notice.',
        'Vaccination appointment has been scheduled.',
        'Clinic visit summary has been uploaded.',
        'Dental cleaning booking reminder.',
        'Health screening report uploaded successfully.',
        'Hospital discharge summary is available.',
        'Follow-up consultation scheduled for next week.',
        'Prescription refill request has been approved.',
        'Urgent health alert: Please contact your doctor.',
        'Eye checkup appointment reminder.',
        'Fitness assessment report is ready.',
        'Lab sample collection scheduled for tomorrow.',
        'Medical bill of Rs.3500 has been generated.',
        'Pathology report indicates further tests required.',
        'Your vaccination certificate is available for download.'
    ],
    'Shopping': [
        'Your order has been shipped.',
        'Order delivered successfully.',
        'Track your parcel using the link provided.',
        'Your shopping cart contains items waiting checkout.',
        'Your return request was approved.',
        'Payment received for order #12345.',
        'Special offer on items in your wishlist.',
        'Thank you for shopping with us.',
        'Your invoice is attached to this email.',
        'Flash sale on selected products today.',
        'New arrivals have just been added.',
        'Product is back in stock.',
        'Cart items are selling out fast.',
        'Limited edition product now available.',
        'Order confirmation number: #98765.',
        'Free delivery on orders above Rs.500.',
        'Exclusive promotion for loyalty members.',
        'Big savings on electronics this weekend.',
        'Subscription box has been dispatched.',
        'Redeem your gift card balance now.'
    ],
    'Travel': [
        'Your flight booking has been confirmed.',
        'Boarding pass is available for download.',
        'Train reservation confirmed with PNR 1234567890.',
        'Hotel reservation confirmed under your name.',
        'Reminder: Online check-in is available for your flight.',
        'Bus ticket booking successful.',
        'Travel insurance policy has been issued.',
        'Flight schedule has been changed by the airline.',
        'Cab booking with driver details attached.',
        'Trip itinerary for your upcoming trip is attached.',
        'Visa appointment confirmation received.',
        'Exclusive discount on holiday packages.',
        'Cruise booking confirmation and details.',
        'Car rental reservation confirmed.',
        'Check-in reminder: Your hotel stay begins today.',
        'Boarding gate information has been updated.',
        'Flight delay notification with revised departure time.',
        'Seat upgrade offer is available for purchase.',
        'Luggage tracking number assigned to your booking.',
        'Travel advisory has been issued for your destination.'
    ],
    'Spam': [
        'Congratulations, you have won a million dollars!',
        'Claim your free iPhone now by clicking here.',
        'This is not a scam; act immediately to claim your reward.',
        'You have been selected for an exclusive inheritance.',
        'Earn thousands from home with zero investment required.',
        'Click here to double your money overnight.',
        'Your account has been compromised; confirm details now.',
        'You are preapproved for a large loan today.',
        'Get paid instantly for completing simple online tasks.',
        'Limited time: Free gift cards available for winners.',
        'Secret crypto investment that guarantees returns.',
        'Your ATM will be blocked; verify now to avoid issues.',
        'Work from home and earn five figures weekly.',
        'Act fast to claim your vacation prize.',
        'Final notice: Pay now to avoid legal action.',
        'Exclusive method to make millions easily.',
        'You are the lucky winner of our contest.',
        'Urgent: Update your payment information immediately.',
        'Free trial that renews at a high cost unless cancelled.',
        'Unlimited free service for a limited period.'
    ],
    'Job Alerts': [
        'New job openings that match your profile.',
        'A recruiter viewed your resume today.',
        'Your job application has been submitted successfully.',
        'Interview invitation scheduled for next week.',
        'New IT positions available in your city.',
        'Your resume has been shortlisted by a company.',
        'Remote job opportunities posted today.',
        'Application status update: Under review.',
        'Apply now to top companies hiring fresh graduates.',
        'Job fair invitation: Register to attend.',
        'Your profile was recommended to several recruiters.',
        'Internship opportunities are now accepting applications.',
        'Reminder: Complete your online application.',
        'Company X has invited you for a screening test.',
        'Walk-in interview scheduled this weekend.',
        'Your candidacy has advanced to the next round.',
        'Offer letter attached for your review.',
        'Background verification process has started.',
        'Training and onboarding schedule has been shared.',
        'Final reminder: Apply before the deadline.'
    ],
    'Technology': [
        'New software update is available for your device.',
        'Security patch released: Please update immediately.',
        'Invitation: Webinar on artificial intelligence.',
        'Discover new programming tutorials this week.',
        'Cybersecurity alert: Protect your account now.',
        'Tech conference registration is now open.',
        'New gadget launch announced today.',
        'Cloud service maintenance scheduled for Sunday.',
        'Device firmware upgrade is available.',
        'Reminder: Update your password for improved security.',
        'AI research paper featured in our newsletter.',
        'Join the developer community forum discussion.',
        'New version of Python has been released.',
        'Workshop on data engineering scheduled next Friday.',
        'Free online course on machine learning available.',
        'Vulnerability report requires immediate action.',
        'IoT device pairing instructions are attached.',
        'Weekly tech digest: Top stories this week.',
        'Beta access granted to new software features.',
        'System reboot required to complete installation.'
    ],
    'Government': [
        'Your Aadhaar update request has been processed.',
        'Income tax filing deadline has been announced.',
        'Voter registration confirmation is available.',
        'Official advisory on public health measures issued.',
        'Notice from the local municipal office.',
        'Passport renewal application status updated.',
        'Subsidy payment has been credited to your account.',
        'Pension scheme enrollment confirmation received.',
        'Public safety advisory issued for your area.',
        'New traffic regulations are now in effect.',
        'Electricity subsidy notification published.',
        'Government scholarship scheme application is open.',
        'RTI response is available for download.',
        'Official election schedule has been published.',
        'Ration card update request has been approved.',
        'GST policy update notification issued.',
        'Public holiday has been declared for your region.',
        'Property tax payment reminder from municipality.',
        'Driving license renewal notice has been issued.',
        'Official gazette notification has been uploaded.'
    ],
    'News & Updates': [
        'Breaking: Global markets show significant movement today.',
        'Your daily newsletter is ready to read.',
        'Market update: Stocks closed higher today.',
        'Company newsletter: Monthly highlights are available.',
        'Sports news: Match results and summaries.',
        'Weather update: Heavy rain expected tomorrow.',
        'Breaking: Election results have been declared.',
        'Weekly magazine issue is now available.',
        'Technology trends update and analysis published.',
        'Entertainment news: New movie releases this week.',
        'Business update: Quarterly earnings announced.',
        'Breaking: Natural disaster reported in the region.',
        'Top headlines: Key stories for today.',
        'Science update: New discovery published.',
        'Local news bulletin for your neighborhood.',
        'Traffic update: Road closures reported today.',
        'Editorial: Opinion piece on recent events.',
        'Health news: Study findings released today.',
        'Weekly digest: Top stories you should not miss.',
        'Market watch: Commodity prices updated.'
    ],
    'Events': [
        'Invitation: Annual conference registration is now open.',
        'Reminder: Webinar on Friday starts at 3 PM.',
        'Your event registration has been confirmed.',
        'Join us for the product launch event tomorrow.',
        'Community meetup scheduled for next Saturday.',
        'Concert tickets are available for purchase.',
        'Leadership workshop invitation: Register today.',
        'Your trade fair pass has been issued.',
        'Live music event: Tickets are selling fast.',
        'Exhibition opens to the public next week.',
        'Hackathon registration confirmation and details.',
        'Invitation: Sports tournament this weekend.',
        'Conference agenda has been updated for attendees.',
        'Art gallery opening: RSVP requested.',
        'Cultural festival event schedule has been released.',
        'Online seminar reminder: Join via the provided link.',
        'Charity fundraiser event details are enclosed.',
        'Photography contest submissions are now open.',
        'Save the date: Startup summit next month.',
        'Award ceremony invitation and RSVP details.'
    ],
    'Legal': [
        'Your contract agreement has been updated.',
        'Privacy policy has been revised and published.',
        'New compliance guidelines have been announced.',
        'Court hearing date has been scheduled.',
        'Legal notice: A response is required within 7 days.',
        'Signed agreement is available for download.',
        'Digital signature verification is pending.',
        'Terms and conditions have been modified.',
        'Consumer rights policy update has been issued.',
        'Arbitration hearing notice has been served.',
        'Contract renewal reminder from legal department.',
        'Non-disclosure agreement requires signing.',
        'Employment agreement document uploaded for review.',
        'Official regulatory filing acknowledgment received.',
        'Copyright claim notice has been filed.',
        'Patent application status update is available.',
        'Tax compliance notice issued by authorities.',
        'Legal advisory from corporate counsel.',
        'E-sign request completed successfully.',
        'Notary appointment confirmation and details.'
    ],
    'Support/Service': [
        'Support ticket #12345 has been received.',
        'Our team has responded to your service request.',
        'Update: Your reported issue has been resolved.',
        'Warranty claim request has been approved.',
        'Need assistance? Contact our support center.',
        'Transcript of your recent chat is attached.',
        'Technical issue reported: We are investigating.',
        'Customer care follow-up scheduled for tomorrow.',
        'Request for replacement has been processed.',
        'Repair appointment scheduled for next week.',
        'Subscription issue has been corrected.',
        'Installation appointment confirmed for your address.',
        'Refund process has been initiated.',
        'Please rate your recent support experience.',
        'Your case has been escalated to the senior team.',
        'Apology for delay: We are working on your request.',
        'Your request remains open and under review.',
        'Patch update is available to fix the reported bug.',
        'Service technician will arrive between 9 AM and 12 PM.',
        'Your support ticket has been closed successfully.'
    ]
}


# -------------------------------
# Generate Synthetic Data
# -------------------------------
def generate_synthetic_data(n=10000):
    data = []
    for _ in range(n):
        label = random.choice(CLASSES)
        subject = random.choice(ALL_PHRASES[label])
        body = random.choice(ALL_PHRASES[label])
        text = f"Subject: {subject} Body: {body}"
        data.append({'text': text, 'label': label})
    return pd.DataFrame(data)

# -------------------------------
# Extract Emails from PDF
# -------------------------------
def extract_emails_from_pdf(pdf_file):
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"

        # Split based on "Subject:" keyword
        emails = re.split(r"Subject:", text, flags=re.IGNORECASE)
        email_list = []
        for e in emails:
            if e.strip():
                lines = e.strip().split("\n", 1)
                subject = lines[0].strip()
                body = lines[1].strip() if len(lines) > 1 else ""
                email_list.append({"subject": subject, "body": body, "full_text": subject + " " + body})

        return pd.DataFrame(email_list) if email_list else None
    except Exception as e:
        st.error(f"Error extracting emails: {e}")
        return None

# -------------------------------
# Main Streamlit App
# -------------------------------
def main():
    st.set_page_config(page_title="📧 Email Classifier", page_icon="📨", layout="wide")

    # Header
    st.title("📧 Email Classifier using Machine Learning")
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #f5f7fa, #c3cfe2);
        }
        .main-title {
            font-size: 28px;
            font-weight: bold;
            color: #2c3e50;
        }
        .sub-title {
            font-size: 18px;
            color: #34495e;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown('<p class="sub-title">Classify emails into categories using ML trained on synthetic data 🚀</p>', unsafe_allow_html=True)

    st.markdown("---")

    # Step 1: Train model
    with st.container():
        st.header("⚙️ Step 1: Train the Model")

        if st.button("Train Classifier Model", use_container_width=True):
            with st.spinner("⏳ Generating synthetic dataset and training model..."):
                df = generate_synthetic_data(10000)
                X_train, X_test, y_train, y_test = train_test_split(
                    df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
                )

                vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
                X_train_vec = vectorizer.fit_transform(X_train)
                X_test_vec = vectorizer.transform(X_test)

                model = LogisticRegression(max_iter=200, n_jobs=-1)
                model.fit(X_train_vec, y_train)

                # Accuracy
                y_train_pred = model.predict(X_train_vec)
                y_test_pred = model.predict(X_test_vec)
                train_acc = accuracy_score(y_train, y_train_pred)*100 
                test_acc = accuracy_score(y_test, y_test_pred)*100

                st.session_state['model'] = model
                st.session_state['vectorizer'] = vectorizer
                st.session_state['train_acc'] = train_acc
                st.session_state['test_acc'] = test_acc
                st.session_state['report'] = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)

            st.success("✅ Model trained successfully!")

            # Metrics
            col1, col2 = st.columns(2)
            col1.metric("Training Accuracy %", f"{train_acc:.2f}")
            col2.metric("Test Accuracy %", f"{test_acc:.2f}")

            # Per-class report
            with st.expander("📊 View Accuracy per Class (Test Data)", expanded=False):
                report = st.session_state['report']
                rep_df = pd.DataFrame(report).transpose().round(2)
                st.dataframe(rep_df.style.background_gradient(cmap="Blues"), use_container_width=True)

    st.markdown("---")

    # Step 2: Upload PDF
    with st.container():
        st.header("📂 Step 2: Upload and Classify Emails")

        if 'model' not in st.session_state:
            st.warning("⚠️ Train the model first before uploading a PDF.")
            return

        uploaded_file = st.file_uploader("Upload a PDF containing emails (Subject & Body)", type="pdf")

        if uploaded_file:
            with st.spinner("🔎 Extracting and classifying emails..."):
                email_df = extract_emails_from_pdf(uploaded_file)
                if email_df is None or email_df.empty:
                    st.error("❌ No emails found in PDF.")
                    return

                model = st.session_state['model']
                vectorizer = st.session_state['vectorizer']
                X_new = vectorizer.transform(email_df['full_text'])
                predictions = model.predict(X_new)
                email_df['Predicted Class'] = predictions

                st.success("✅ Classification complete!")

                # Distribution chart
                counts = Counter(predictions)
                counts_df = pd.DataFrame(counts.items(), columns=["Class", "Count"]).set_index("Class").reindex(CLASSES).fillna(0).reset_index()
                
                fig = px.bar(
                    counts_df, x="Class", y="Count", text="Count",
                    color="Class", title="📊 Distribution of Predicted Categories"
                )
                st.plotly_chart(fig, use_container_width=True)

                # Results Table
                with st.expander("🔍 View Classified Emails", expanded=True):
                    st.dataframe(email_df[['subject', 'Predicted Class']].head(50), use_container_width=True)


if __name__ == "__main__":
    main()

